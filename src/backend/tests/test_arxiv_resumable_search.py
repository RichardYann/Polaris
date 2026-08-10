"""Regressions for shared arXiv query cooldown and resumable paging."""

import fakeredis.aioredis
import httpx
import pytest
import respx

from app.services.literature.arxiv import ArxivClient, ArxivRateLimitedError


@respx.mock
async def test_arxiv_429_sets_global_query_cooldown():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client = ArxivClient(redis=redis, min_interval=0, max_retries=1, backoff_base=0)
    route = respx.get("https://export.arxiv.org/api/query").mock(return_value=httpx.Response(429))
    with pytest.raises(ArxivRateLimitedError) as first:
        await client.search_page(keywords=["agents"], limit=1)
    assert first.value.retry_after == 600
    with pytest.raises(ArxivRateLimitedError) as second:
        await client.search_page(keywords=["agents"], limit=1)
    assert second.value.retry_after and second.value.retry_after > 0
    assert route.call_count == 1
    await redis.aclose()
