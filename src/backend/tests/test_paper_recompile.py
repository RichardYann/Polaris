"""Regressions for durable paper recompile locking."""

import uuid

import fakeredis.aioredis

from app.services.paper_recompile import launch_recompile
from tests.conftest import StubQueue


async def test_recompile_lock_reuses_same_users_task_and_rejects_other_user():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = StubQueue()
    paper_id = uuid.uuid4()
    owner = uuid.uuid4()
    first = await launch_recompile(redis=redis, queue=queue, paper_id=paper_id, user_id=owner)
    again = await launch_recompile(redis=redis, queue=queue, paper_id=paper_id, user_id=owner)
    other = await launch_recompile(
        redis=redis, queue=queue, paper_id=paper_id, user_id=uuid.uuid4()
    )
    assert first and again == first
    assert other is None
    assert len(queue.jobs) == 1
    await redis.aclose()
