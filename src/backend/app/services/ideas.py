"""Idea Forge / 评审锦标赛业务逻辑（不 import fastapi）。"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.models.gate import Gate
from app.models.idea import IDEA_STATUSES, Idea
from app.models.library_direction import LibraryPaper
from app.models.paper import Concept, Paper, paper_concepts
from app.models.project import Project, ProjectMember
from app.models.review import ReviewSession
from app.models.user import User
from app.models.voyage import TERMINAL_STATUSES, VoyageRun
from app.schemas.idea import DeepIdeaRequest, ForgeKnobs
from app.schemas.review import TournamentRequest
from app.services.concepts import library_concept_ids
from app.services.libraries import get_source_library_ids
from app.services.projects import in_my_projects

# 同项目批量 idea 任务互斥；深度生成独立限流，允许不同种子并行。
IDEA_VOYAGE_KINDS = ("idea_forge", "idea_review", "idea_proposal")
EXCLUSIVE_IDEA_VOYAGE_KINDS = ("idea_forge", "idea_review")
MAX_CONCURRENT_DEEP_VOYAGES = 4

# 预算从 knobs 派生：每个候选 idea 预留的 token 额度（gap 分析+生成+打分+去重）
_TOKENS_PER_IDEA = 20_000
# 每场辩论每次 LLM 调用预留：辩论上下文逐轮累积（双方发言+人设+历史），裁判看全场，
# 思考型模型下单次可达 15-20k，估低会在汇总前触发预算门（改由 §5.4 降级收尾兜底）
_TOKENS_PER_MATCH_CALL = 16_000
# 深耕 voyage 默认预算（目标构建工具循环 + 各节起草 + 评审修订）
_DEEP_DEFAULT_BUDGET = 400_000

IDEA_SORTS = ("elo", "-created_at", "score")

# 深耕相关闸门（docs/api-idea2.md §4/§5）
DEEP_GATE_KINDS = ("idea_goal", "idea_pivot")


class IdeaVoyageConflictError(Exception):
    """同一项目已有互斥的 forge/review voyage 在跑。"""


class DeepVoyageLimitError(Exception):
    """同一项目进行中的深度生成已达到上限。"""


class DeepSeedConflictError(Exception):
    """同一种子已有深度生成任务尚未结束。"""


class NotEnoughIdeasError(Exception):
    """锦标赛参与 idea 不足两个。"""


class InvalidIdeaIdsError(Exception):
    """显式 idea_ids 含不存在/不属于本项目的 id。"""


class InvalidSeedError(Exception):
    """深耕种子引用的 concept/paper/idea 不存在或不属于本项目。"""


class TournamentRetryUnavailableError(Exception):
    """The latest tournament is running, fully successful, or already undone."""


class TournamentUndoUnavailableError(Exception):
    """The latest tournament cannot be safely undone."""


# ---- voyage 创建 ----


async def find_running_idea_voyage(
    session: AsyncSession, project_id: uuid.UUID
) -> VoyageRun | None:
    """同项目正在运行的互斥批量任务（forge/review，不含 proposal）。"""
    stmt = (
        select(VoyageRun)
        .where(
            VoyageRun.project_id == project_id,
            VoyageRun.kind.in_(EXCLUSIVE_IDEA_VOYAGE_KINDS),
            VoyageRun.status.not_in(tuple(TERMINAL_STATUSES)),
        )
        .order_by(VoyageRun.created_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _lock_project_idea_creation(session: AsyncSession, project_id: uuid.UUID) -> None:
    """串行化同项目的任务创建，避免并发请求同时越过计数/互斥检查。"""
    await session.execute(select(Project.id).where(Project.id == project_id).with_for_update())


async def _running_deep_voyages(
    session: AsyncSession, project_id: uuid.UUID
) -> list[VoyageRun]:
    stmt = (
        select(VoyageRun)
        .where(
            VoyageRun.project_id == project_id,
            VoyageRun.kind == "idea_proposal",
            VoyageRun.status.not_in(tuple(TERMINAL_STATUSES)),
        )
        .order_by(VoyageRun.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def create_forge_voyage(
    session: AsyncSession,
    *,
    project: Project,
    knobs: ForgeKnobs,
    created_by: uuid.UUID | None,
) -> VoyageRun:
    """建 idea_forge voyage（forge/review 互斥 + Activity 落记录），由调用方入队 run_voyage。"""
    await _lock_project_idea_creation(session, project.id)
    if await find_running_idea_voyage(session, project.id) is not None:
        raise IdeaVoyageConflictError(str(project.id))
    run = VoyageRun(
        kind="idea_forge",
        goal=f"Idea Forge：{project.name}",
        status="planning",
        cursor=0,
        checkpoint={"params": {"knobs": knobs.model_dump()}},
        budget={"max_tokens": int(knobs.num_ideas) * _TOKENS_PER_IDEA},
        project_id=project.id,
        created_by=created_by,
    )
    session.add(run)
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="forge.started",
            message=f"Idea Forge 已启动（目标 {knobs.num_ideas} 个候选）",
            payload={"knobs": knobs.model_dump()},
        )
    )
    await session.commit()
    await session.refresh(run)
    return run


async def create_tournament_voyage(
    session: AsyncSession,
    *,
    project: Project,
    data: TournamentRequest,
    created_by: uuid.UUID | None,
) -> VoyageRun:
    """建 idea_review（辩论锦标赛）voyage；参与者不足 2 个抛 NotEnoughIdeasError。"""
    await _lock_project_idea_creation(session, project.id)
    if await find_running_idea_voyage(session, project.id) is not None:
        raise IdeaVoyageConflictError(str(project.id))

    if data.idea_ids:
        wanted = list(dict.fromkeys(data.idea_ids))
        stmt = select(Idea.id).where(
            Idea.project_id == project.id,
            Idea.id.in_(wanted),
            Idea.trashed_at.is_(None),
        )
        found = {row for (row,) in (await session.execute(stmt)).all()}
        missing = [str(i) for i in wanted if i not in found]
        if missing:
            raise InvalidIdeaIdsError(", ".join(missing))
        participant_count = len(wanted)
    else:
        stmt = select(func.count()).where(
            Idea.project_id == project.id,
            Idea.status.in_(("candidate", "under_review")),
            Idea.trashed_at.is_(None),
        )
        participant_count = int((await session.execute(stmt)).scalar_one())
    if participant_count < 2:
        raise NotEnoughIdeasError(str(project.id))

    matches = participant_count // 2
    params: dict[str, Any] = {
        "idea_ids": [str(i) for i in data.idea_ids] if data.idea_ids else None,
        "rounds": data.rounds,
        "personas": [p.model_dump() for p in data.personas] if data.personas else None,
    }
    run = VoyageRun(
        kind="idea_review",
        goal=f"Idea 评审锦标赛：{project.name}",
        status="planning",
        cursor=0,
        checkpoint={"params": params},
        budget={"max_tokens": matches * (2 * data.rounds + 1) * _TOKENS_PER_MATCH_CALL},
        project_id=project.id,
        created_by=created_by,
    )
    session.add(run)
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="review.started",
            message=f"Idea 评审锦标赛已启动（{participant_count} 个想法）",
            payload=params,
        )
    )
    await session.commit()
    await session.refresh(run)
    return run


async def reset_tournament_ratings(
    session: AsyncSession,
    *,
    project: Project,
    created_by: uuid.UUID | None,
) -> int:
    """Reset Elo and win/loss counters for all non-trashed ideas in a project."""
    await _lock_project_idea_creation(session, project.id)
    if await find_running_idea_voyage(session, project.id) is not None:
        raise IdeaVoyageConflictError(str(project.id))

    ideas = list(
        (
            await session.execute(
                select(Idea).where(
                    Idea.project_id == project.id,
                    Idea.trashed_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    for idea in ideas:
        idea.elo_rating = 1200.0
        idea.matches = 0
        idea.wins = 0
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="review.ratings_reset",
            message=f"Idea 锦标赛评分已重置（{len(ideas)} 个当前想法）",
            payload={"affected": len(ideas), "scope": "active"},
        )
    )
    await session.commit()
    return len(ideas)


def _review_pair_key(idea_a: str, idea_b: str) -> tuple[str, str]:
    return (idea_a, idea_b) if idea_a < idea_b else (idea_b, idea_a)


def _failed_review_pairs(run: VoyageRun) -> list[list[str]]:
    checkpoint = run.checkpoint or {}
    planned = checkpoint.get("review_pairs") or []
    results = checkpoint.get("review_results") or []
    completed = {
        _review_pair_key(str(r.get("idea_a")), str(r.get("idea_b")))
        for r in results
        if isinstance(r, dict) and r.get("idea_a") and r.get("idea_b")
    }
    return [
        [str(pair[0]), str(pair[1])]
        for pair in planned
        if isinstance(pair, list)
        and len(pair) == 2
        and _review_pair_key(str(pair[0]), str(pair[1])) not in completed
    ]


async def latest_review_tournament(
    session: AsyncSession, project_id: uuid.UUID
) -> VoyageRun | None:
    stmt = (
        select(VoyageRun)
        .where(VoyageRun.project_id == project_id, VoyageRun.kind == "idea_review")
        .order_by(VoyageRun.created_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def review_tournament_summary(run: VoyageRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    checkpoint = run.checkpoint or {}
    planned = checkpoint.get("review_pairs") or []
    results = checkpoint.get("review_results") or []
    failed = _failed_review_pairs(run)
    undone = bool(checkpoint.get("review_undone_at"))
    params = checkpoint.get("params") or {}
    return {
        "voyage_id": str(run.id),
        "root_voyage_id": str(params.get("retry_of") or run.id),
        "status": run.status,
        "planned": len(planned),
        "completed": len(results),
        "failed": len(failed),
        "is_retry": bool(params.get("retry_of")),
        "undone": undone,
        "can_retry": run.status in TERMINAL_STATUSES and bool(failed) and not undone,
        "can_undo": run.status in TERMINAL_STATUSES and not undone,
    }


async def create_retry_tournament_voyage(
    session: AsyncSession,
    *,
    project: Project,
    source: VoyageRun,
    created_by: uuid.UUID | None,
) -> VoyageRun:
    """Create a supplement run containing only the source run's unfinished pairs."""
    await _lock_project_idea_creation(session, project.id)
    if await find_running_idea_voyage(session, project.id) is not None:
        raise IdeaVoyageConflictError(str(project.id))
    if source.project_id != project.id or source.kind != "idea_review":
        raise TournamentRetryUnavailableError(str(source.id))
    if source.status not in TERMINAL_STATUSES or (source.checkpoint or {}).get("review_undone_at"):
        raise TournamentRetryUnavailableError(str(source.id))
    failed_pairs = _failed_review_pairs(source)
    if not failed_pairs:
        raise TournamentRetryUnavailableError(str(source.id))

    source_params = (source.checkpoint or {}).get("params") or {}
    root_id = str(source_params.get("retry_of") or source.id)
    params = {
        "idea_ids": sorted({idea_id for pair in failed_pairs for idea_id in pair}),
        "rounds": int(source_params.get("rounds") or 2),
        "personas": source_params.get("personas"),
        "retry_pairs": failed_pairs,
        "retry_of": root_id,
        "retry_source": str(source.id),
    }
    matches = len(failed_pairs)
    run = VoyageRun(
        kind="idea_review",
        goal=f"Idea 评审锦标赛补赛：{project.name}",
        status="planning",
        cursor=0,
        checkpoint={"params": params},
        budget={
            "max_tokens": matches
            * (2 * int(params["rounds"]) + 1)
            * _TOKENS_PER_MATCH_CALL
        },
        project_id=project.id,
        created_by=created_by,
    )
    session.add(run)
    await session.flush()
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="review.retry_started",
            message=f"Idea 锦标赛补赛已启动（{matches} 场待补）",
            payload={
                "voyage_id": str(run.id),
                "root_voyage_id": root_id,
                "source_voyage_id": str(source.id),
                "matches": matches,
            },
        )
    )
    await session.commit()
    await session.refresh(run)
    return run


def _reverse_elo(rating_a: float, rating_b: float, winner: str) -> tuple[float, float]:
    """Recover pre-match Elo from the two post-match ratings (K=32, no draws)."""
    total = rating_a + rating_b
    post_diff = rating_a - rating_b
    score_a = 1.0 if winner == "a" else 0.0
    low, high = -8000.0, 8000.0
    for _ in range(100):
        diff = (low + high) / 2.0
        expected_a = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        calculated_post = diff + 64.0 * (score_a - expected_a)
        if calculated_post < post_diff:
            low = diff
        else:
            high = diff
    before_diff = (low + high) / 2.0
    return (total + before_diff) / 2.0, (total - before_diff) / 2.0


async def undo_latest_tournament(
    session: AsyncSession,
    *,
    project: Project,
    created_by: uuid.UUID | None,
) -> int:
    """Undo the latest logical tournament round while preserving task/cost audit rows."""
    await _lock_project_idea_creation(session, project.id)
    if await find_running_idea_voyage(session, project.id) is not None:
        raise IdeaVoyageConflictError(str(project.id))
    latest = await latest_review_tournament(session, project.id)
    if latest is None or latest.status not in TERMINAL_STATUSES:
        raise TournamentUndoUnavailableError(str(project.id))
    latest_params = (latest.checkpoint or {}).get("params") or {}
    root_id = str(latest_params.get("retry_of") or latest.id)

    runs = list(
        (
            await session.execute(
                select(VoyageRun)
                .where(VoyageRun.project_id == project.id, VoyageRun.kind == "idea_review")
                .order_by(VoyageRun.created_at)
            )
        )
        .scalars()
        .all()
    )
    group = [
        run
        for run in runs
        if str(run.id) == root_id
        or str(((run.checkpoint or {}).get("params") or {}).get("retry_of") or "") == root_id
    ]
    if not group or any((run.checkpoint or {}).get("review_undone_at") for run in group):
        raise TournamentUndoUnavailableError(root_id)
    group_ids = {str(run.id) for run in group}
    root = next((run for run in group if str(run.id) == root_id), None)
    if root is None:
        raise TournamentUndoUnavailableError(root_id)

    sessions = list(
        (
            await session.execute(
                select(ReviewSession)
                .join(Idea, Idea.id == ReviewSession.target_id)
                .where(
                    Idea.project_id == project.id,
                    ReviewSession.target_type == "idea_match",
                )
                .order_by(ReviewSession.created_at)
            )
        )
        .scalars()
        .all()
    )
    all_planned = {
        _review_pair_key(str(pair[0]), str(pair[1]))
        for run in group
        for pair in ((run.checkpoint or {}).get("review_pairs") or [])
        if isinstance(pair, list) and len(pair) == 2
    }
    end_at = max(run.updated_at for run in group)
    selected_sessions = []
    for review_session in sessions:
        payload = review_session.payload or {}
        voyage_id = str(payload.get("voyage_id") or "")
        pair = _review_pair_key(
            str(payload.get("idea_a") or review_session.target_id),
            str(payload.get("idea_b") or ""),
        )
        legacy_match = (
            not voyage_id
            and root.created_at <= review_session.created_at <= end_at
            and pair in all_planned
        )
        if voyage_id in group_ids or legacy_match:
            selected_sessions.append(review_session)

    snapshot = (root.checkpoint or {}).get("review_standings_before")
    if isinstance(snapshot, dict) and snapshot:
        for idea_id, before in snapshot.items():
            idea = await session.get(Idea, uuid.UUID(str(idea_id)))
            if idea is None or not isinstance(before, dict):
                continue
            idea.elo_rating = float(before.get("elo_rating", 1200.0))
            idea.matches = int(before.get("matches", 0))
            idea.wins = int(before.get("wins", 0))
            if before.get("status"):
                idea.status = str(before["status"])
    else:
        # Legacy rounds did not snapshot standings. Because every idea appears at most
        # once per logical round, the Elo transform is safely invertible.
        for review_session in reversed(selected_sessions):
            payload = review_session.payload or {}
            winner = payload.get("winner")
            if review_session.status != "closed" or winner not in ("a", "b"):
                continue
            idea_a = await session.get(Idea, review_session.target_id)
            idea_b_raw = payload.get("idea_b")
            idea_b = await session.get(Idea, uuid.UUID(str(idea_b_raw))) if idea_b_raw else None
            if idea_a is None or idea_b is None:
                continue
            idea_a.elo_rating, idea_b.elo_rating = _reverse_elo(
                idea_a.elo_rating, idea_b.elo_rating, str(winner)
            )
            idea_a.matches = max(0, idea_a.matches - 1)
            idea_b.matches = max(0, idea_b.matches - 1)
            winner_idea = idea_a if winner == "a" else idea_b
            winner_idea.wins = max(0, winner_idea.wins - 1)

    for review_session in selected_sessions:
        await session.delete(review_session)
    undone_at = datetime.now(UTC).isoformat()
    for run in group:
        checkpoint = dict(run.checkpoint or {})
        checkpoint["review_undone_at"] = undone_at
        checkpoint["review_results"] = []
        checkpoint["review_failed_pairs"] = []
        run.checkpoint = checkpoint
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="review.round_undone",
            message=f"已撤销最新一轮 Idea 锦标赛（删除 {len(selected_sessions)} 场记录）",
            payload={
                "root_voyage_id": root_id,
                "voyage_ids": sorted(group_ids),
                "sessions": len(selected_sessions),
            },
        )
    )
    await session.commit()
    return len(selected_sessions)


async def _validate_seed(
    session: AsyncSession, *, project_id: uuid.UUID, seed_type: str, value: str
) -> str:
    """引用型种子存在性校验，返回种子摘要（写入 voyage goal 文案）。"""
    if seed_type == "text":
        return value[:80]
    try:
        target_id = uuid.UUID(value)
    except ValueError as e:
        raise InvalidSeedError(value) from e
    if seed_type == "paper":
        paper = await session.get(Paper, target_id)
        if paper is None:
            raise InvalidSeedError(value)
        library_ids = await get_source_library_ids(session, project_id)
        in_corpus = bool(library_ids) and (
            await session.execute(
                select(LibraryPaper.paper_id)
                .where(
                    LibraryPaper.library_id.in_(library_ids),
                    LibraryPaper.paper_id == paper.id,
                )
                .limit(1)
            )
        ).first() is not None
        if not in_corpus:
            raise InvalidSeedError(value)
        return f"论文《{paper.title[:60]}》"
    if seed_type == "concept":
        concept = await session.get(Concept, target_id)
        if concept is None:
            raise InvalidSeedError(value)
        library_ids = await get_source_library_ids(session, project_id)
        # 概念不属于任何库：在不在本课题语料内 = 有没有本课题库里的论文用到它
        in_corpus = bool(library_ids) and (
            await session.execute(
                library_concept_ids(library_ids)
                .where(paper_concepts.c.concept_id == concept.id)
                .limit(1)
            )
        ).first() is not None
        if not in_corpus:
            raise InvalidSeedError(value)
        return f"概念「{concept.name}」"
    if seed_type == "idea":
        idea = await session.get(Idea, target_id)
        if idea is None or idea.project_id != project_id:
            raise InvalidSeedError(value)
        return f"草案「{idea.title[:60]}」"
    raise InvalidSeedError(seed_type)


async def create_deep_voyage(
    session: AsyncSession,
    *,
    project: Project,
    data: DeepIdeaRequest,
    created_by: uuid.UUID | None,
) -> VoyageRun:
    """建 idea_proposal voyage（深度生成，docs/api-idea2.md §2），由调用方入队 run_voyage。"""
    await _lock_project_idea_creation(session, project.id)
    running = await _running_deep_voyages(session, project.id)
    if len(running) >= MAX_CONCURRENT_DEEP_VOYAGES:
        raise DeepVoyageLimitError(str(project.id))
    seed = data.seed.model_dump()
    if any(((run.checkpoint or {}).get("params") or {}).get("seed") == seed for run in running):
        raise DeepSeedConflictError(str(project.id))
    seed_brief = await _validate_seed(
        session, project_id=project.id, seed_type=data.seed.type, value=data.seed.value
    )
    budget = data.knobs.budget_tokens or _DEEP_DEFAULT_BUDGET
    run = VoyageRun(
        kind="idea_proposal",
        goal=f"深度研究方案：{project.name}",
        status="planning",
        cursor=0,
        checkpoint={"params": {"seed": data.seed.model_dump(), "knobs": data.knobs.model_dump()}},
        budget={"max_tokens": budget},
        project_id=project.id,
        created_by=created_by,
    )
    session.add(run)
    session.add(
        Activity(
            project_id=project.id,
            actor=f"user:{created_by}" if created_by else "system",
            kind="idea.deep_started",
            message=f"深度想法生成已启动（种子：{seed_brief}）",
            payload={"seed": data.seed.model_dump(), "knobs": data.knobs.model_dump()},
        )
    )
    await session.commit()
    await session.refresh(run)
    return run


async def deep_state(session: AsyncSession, project: Project) -> dict[str, Any]:
    """深度生成状态：最多四条运行中 voyage，各自携带种子与待审批闸门。"""
    running = await _running_deep_voyages(session, project.id)
    running_ids = {str(run.id) for run in running}
    gate_by_run: dict[str, uuid.UUID] = {}
    if running_ids:
        gate_stmt = (
            select(Gate)
            .where(
                Gate.project_id == project.id,
                Gate.kind.in_(DEEP_GATE_KINDS),
                Gate.status == "pending",
            )
            .order_by(Gate.created_at.desc())
        )
        for gate in (await session.execute(gate_stmt)).scalars().all():
            voyage_id = str((gate.payload or {}).get("voyage_id") or "")
            if voyage_id in running_ids and voyage_id not in gate_by_run:
                gate_by_run[voyage_id] = gate.id

    running_voyages = []
    for run in running:
        params = ((run.checkpoint or {}).get("params") or {})
        seed = params.get("seed") if isinstance(params.get("seed"), dict) else None
        running_voyages.append(
            {
                "voyage_id": run.id,
                "status": run.status,
                "seed": seed,
                "pending_gate_id": gate_by_run.get(str(run.id)),
            }
        )

    last_stmt = (
        select(VoyageRun)
        .where(VoyageRun.project_id == project.id, VoyageRun.kind == "idea_proposal")
        .order_by(VoyageRun.created_at.desc())
        .limit(1)
    )
    last = (await session.execute(last_stmt)).scalar_one_or_none()
    last_run: dict[str, Any] | None = None
    if last is not None:
        last_run = {
            "voyage_id": last.id,
            "status": last.status,
            "finished_at": last.updated_at if last.status in TERMINAL_STATUSES else None,
        }
    return {
        # 旧字段保留一个发布周期，兼容尚未升级的前端/桌面客户端。
        "running_voyage_id": running[0].id if running else None,
        "pending_gate_id": gate_by_run.get(str(running[0].id)) if running else None,
        "running_voyages": running_voyages,
        "max_concurrent": MAX_CONCURRENT_DEEP_VOYAGES,
        "last_run": last_run,
    }


# ---- forge 状态 ----


async def idea_counts(session: AsyncSession, project_id: uuid.UUID) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Idea.status, func.count())
            .where(Idea.project_id == project_id, Idea.trashed_at.is_(None))
            .group_by(Idea.status)
        )
    ).all()
    counts = {status: 0 for status in IDEA_STATUSES}
    total = 0
    for status, count in rows:
        counts[status] = int(count)
        total += int(count)
    counts["total"] = total
    return counts


async def forge_state(session: AsyncSession, project: Project) -> dict[str, Any]:
    running = await find_running_idea_voyage(session, project.id)
    stmt = (
        select(VoyageRun)
        .where(VoyageRun.project_id == project.id, VoyageRun.kind == "idea_forge")
        .order_by(VoyageRun.created_at.desc())
        .limit(1)
    )
    last = (await session.execute(stmt)).scalar_one_or_none()
    last_run: dict[str, Any] | None = None
    if last is not None:
        last_run = {
            "voyage_id": last.id,
            "status": last.status,
            "finished_at": last.updated_at if last.status in TERMINAL_STATUSES else None,
        }
    return {
        "running_voyage_id": running.id if running else None,
        "last_run": last_run,
        "idea_counts": await idea_counts(session, project.id),
    }


# ---- idea 读写 ----


def composite_score(idea: Idea) -> float:
    scores = idea.scores if isinstance(idea.scores, dict) else {}
    values = [float(v) for v in scores.values() if isinstance(v, int | float)]
    return sum(values) / len(values) if values else -1.0


async def list_ideas(
    session: AsyncSession,
    *,
    project_ids: Sequence[uuid.UUID],
    status: str | None = None,
    depth: str | None = None,
    research_type: str | None = None,
    sort: str = "-created_at",
    trashed: bool = False,
) -> list[Idea]:
    """这些课题下的想法（列表而非单个 id 的理由见 experiments.list_experiments）。"""
    if not project_ids:
        return []
    trash_cond = Idea.trashed_at.is_not(None) if trashed else Idea.trashed_at.is_(None)
    stmt = select(Idea).where(Idea.project_id.in_(project_ids), trash_cond)
    if trashed:
        return list((await session.execute(stmt.order_by(Idea.trashed_at.desc()))).scalars().all())
    if status:
        stmt = stmt.where(Idea.status == status)
    if depth:
        stmt = stmt.where(Idea.depth == depth)
    if research_type:
        stmt = stmt.where(Idea.research_type == research_type)
    if sort == "elo":
        stmt = stmt.order_by(Idea.elo_rating.desc(), Idea.created_at.desc())
    else:
        stmt = stmt.order_by(Idea.created_at.desc())
    ideas = list((await session.execute(stmt)).scalars().all())
    if sort == "score":  # composite 存 JSON，跨方言排序在 Python 侧做
        ideas.sort(key=composite_score, reverse=True)
    return ideas


async def get_idea_for_user(
    session: AsyncSession, *, idea_id: uuid.UUID, user_id: uuid.UUID
) -> Idea | None:
    """取 idea；非项目成员视为不存在（平台管理员够得着全部课题）。"""
    stmt = select(Idea).where(
        Idea.id == idea_id, in_my_projects(Idea.project_id, user_id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _owned_ideas(
    session: AsyncSession, *, project_id: uuid.UUID, ids: list[uuid.UUID]
) -> list[Idea]:
    if not ids:
        return []
    stmt = select(Idea).where(Idea.project_id == project_id, Idea.id.in_(ids))
    return list((await session.execute(stmt)).scalars().all())


async def trash_ideas(session: AsyncSession, *, project_id: uuid.UUID, ids: list[uuid.UUID]) -> int:
    """移入回收站（软删除）；返回受影响数量。"""
    now = datetime.now(UTC)
    n = 0
    for idea in await _owned_ideas(session, project_id=project_id, ids=ids):
        if idea.trashed_at is None:
            idea.trashed_at = now
            n += 1
    await session.commit()
    return n


async def restore_ideas(
    session: AsyncSession, *, project_id: uuid.UUID, ids: list[uuid.UUID]
) -> int:
    n = 0
    for idea in await _owned_ideas(session, project_id=project_id, ids=ids):
        if idea.trashed_at is not None:
            idea.trashed_at = None
            n += 1
    await session.commit()
    return n


async def purge_ideas(
    session: AsyncSession, *, project_id: uuid.UUID, ids: list[uuid.UUID] | None = None
) -> int:
    """永久删除。ids=None → 清空该项目回收站；否则只删指定 id 中已在回收站的。
    级联删除其实验（DB FK ondelete=CASCADE）。返回删除数量。"""
    if ids is None:
        rows = await list_ideas(session, project_ids=[project_id], trashed=True)
    else:
        rows = [
            i
            for i in await _owned_ideas(session, project_id=project_id, ids=ids)
            if i.trashed_at is not None
        ]
    n = len(rows)
    for idea in rows:
        await session.delete(idea)
    await session.commit()
    return n


async def set_idea_status(session: AsyncSession, idea: Idea, status: str) -> Idea:
    idea.status = status
    await session.commit()
    await session.refresh(idea)
    return idea


def parse_parent_ids(idea: Idea) -> list[uuid.UUID]:
    """idea.parent_paper_ids（JSON 字符串列表）→ UUID 列表（无效 id 静默忽略）。"""
    ids: list[uuid.UUID] = []
    for raw in idea.parent_paper_ids or []:
        try:
            ids.append(uuid.UUID(str(raw)))
        except ValueError:
            continue
    return ids


async def parent_papers_brief(session: AsyncSession, idea: Idea) -> list[dict[str, Any]]:
    """IdeaDetail.parent_papers：{id, title}（已删除的论文静默忽略）。"""
    ids = parse_parent_ids(idea)
    if not ids:
        return []
    rows = (await session.execute(select(Paper.id, Paper.title).where(Paper.id.in_(ids)))).all()
    by_id = {pid: title for pid, title in rows}
    return [{"id": pid, "title": by_id[pid]} for pid in ids if pid in by_id]


async def seed_idea_brief(session: AsyncSession, idea: Idea) -> dict[str, Any] | None:
    """IdeaDetail.seed_idea：深化来源草案 {id, title}（已删除静默为 None）。"""
    if idea.seed_idea_id is None:
        return None
    seed = await session.get(Idea, idea.seed_idea_id)
    return {"id": seed.id, "title": seed.title} if seed is not None else None


# ---- 晋级（idea_promotion 闸门） ----


async def can_promote(session: AsyncSession, *, project_id: uuid.UUID, user: User) -> bool:
    """promote 权限：项目 owner（成员角色 owner）或平台 admin。"""
    if user.role == "admin":
        return True
    member = await session.get(ProjectMember, (project_id, user.id))
    return member is not None and member.role == "owner"


async def find_pending_promotion_gate(session: AsyncSession, idea_id: uuid.UUID) -> Gate | None:
    stmt = select(Gate).where(
        Gate.kind == "idea_promotion",
        Gate.status == "pending",
        Gate.payload["idea_id"].as_string() == str(idea_id),
    )
    return (await session.execute(stmt)).scalars().first()


async def create_promotion_gate(session: AsyncSession, idea: Idea, user: User) -> Gate:
    gate = Gate(
        project_id=idea.project_id,
        kind="idea_promotion",
        payload={"idea_id": str(idea.id), "idea_title": idea.title},
        requested_by=f"user:{user.id}",
    )
    session.add(gate)
    session.add(
        Activity(
            project_id=idea.project_id,
            actor=f"user:{user.id}",
            kind="idea.promote_requested",
            message=f"想法「{idea.title}」申请晋级，等待人工审批",
            payload={"idea_id": str(idea.id), "gate_id": None},
        )
    )
    await session.commit()
    await session.refresh(gate)
    return gate


async def promote_from_gate(session: AsyncSession, gate: Gate) -> Idea | None:
    """gates approve 联动：payload.idea_id 存在则把 idea 置为 promoted，返回该 idea。"""
    raw = (gate.payload or {}).get("idea_id")
    if not raw:
        return None
    try:
        idea_id = uuid.UUID(str(raw))
    except ValueError:
        return None
    idea = await session.get(Idea, idea_id)
    if idea is None:
        return None
    if idea.status != "promoted":
        idea.status = "promoted"
        session.add(
            Activity(
                project_id=idea.project_id,
                actor=f"user:{gate.decided_by}" if gate.decided_by else "system",
                kind="idea.promoted",
                message=f"想法「{idea.title}」已通过审批晋级",
                payload={"idea_id": str(idea.id), "gate_id": str(gate.id)},
            )
        )
        await session.commit()
        await session.refresh(idea)
    return idea
