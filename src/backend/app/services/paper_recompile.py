"""Durable background orchestration for single-paper wiki recompilation."""

import asyncio
import logging
import uuid

from redis.asyncio import Redis

from app.core.db import get_sessionmaker
from app.core.events import EventBus, publish_paper_task_event
from app.core.queue import TaskQueue
from app.services import papers as papers_service
from app.services import wiki_compile as wiki_compile_service
from app.services.paper_enrich import paper_task_owner_key

logger = logging.getLogger(__name__)

TASK_TTL_SECONDS = 2 * 3600
_WORKER_COMPILE_SLOTS = asyncio.Semaphore(2)


def compile_lock_key(paper_id: uuid.UUID | str) -> str:
    return f"paper_compile_lock:{paper_id}"


async def launch_recompile(
    *, redis: Redis, queue: TaskQueue, paper_id: uuid.UUID, user_id: uuid.UUID
) -> str | None:
    task_id = uuid.uuid4().hex
    locked = await redis.set(compile_lock_key(paper_id), task_id, ex=TASK_TTL_SECONDS, nx=True)
    if not locked:
        existing = await redis.get(compile_lock_key(paper_id))
        if isinstance(existing, bytes):
            existing = existing.decode()
        if existing:
            owner = await redis.get(paper_task_owner_key(str(existing)))
            if isinstance(owner, bytes):
                owner = owner.decode()
            if owner == str(user_id):
                return str(existing)
        return None
    try:
        await redis.setex(paper_task_owner_key(task_id), TASK_TTL_SECONDS, str(user_id))
        await queue.enqueue("recompile_paper_task", task_id, str(paper_id), str(user_id))
    except Exception:
        if await redis.get(compile_lock_key(paper_id)) in (task_id, task_id.encode()):
            await redis.delete(compile_lock_key(paper_id))
        raise
    return task_id


async def run_recompile_task(
    *, redis: Redis, task_id: str, paper_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    bus = EventBus(redis)

    async def emit(stage: str, status: str, detail: str | None = None) -> None:
        await publish_paper_task_event(
            bus, task_id, "stage", {"stage": stage, "status": status, "detail": detail}
        )

    try:
        async with _WORKER_COMPILE_SLOTS, get_sessionmaker()() as session:
            view = await papers_service.get_paper_for_user(
                session,
                paper_id=paper_id,
                user_id=user_id,
                with_concepts=True,
            )
            if view is None:
                raise LookupError("论文已删除或当前用户已无权访问")
            await wiki_compile_service.recompile_paper(session, view, user_id=user_id, emit=emit)
        await publish_paper_task_event(bus, task_id, "done", {})
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("paper recompile task failed: %s", task_id)
        await publish_paper_task_event(
            bus, task_id, "error", {"message": f"{type(e).__name__}: {e}"}
        )
    finally:
        key = compile_lock_key(paper_id)
        if await redis.get(key) in (task_id, task_id.encode()):
            await redis.delete(key)
