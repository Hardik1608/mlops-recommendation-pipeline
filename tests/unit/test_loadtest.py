"""The load generator produces the published latency numbers, so its parsing is tested."""

from __future__ import annotations

import asyncio

import pytest

from recsys.loadtest import (
    KeepAlivePool,
    histogram_quantile,
    parse_histogram,
    percentiles,
    read_response,
)

METRIC = "recsys_http_request_duration_seconds_bucket"
REC = 'route="/recommendations/{user_id}"'
BUCKETS = "\n".join(
    [
        "# HELP recsys_http_request_duration_seconds latency",
        f'{METRIC}{{le="0.1",{REC},variant="champion"}} 0.0',
        f'{METRIC}{{{REC},variant="challenger",le="0.25"}} 40.0',  # label order varies
        f'{METRIC}{{le="0.25",{REC},variant="champion"}} 60.0',
        f'{METRIC}{{le="+Inf",{REC},variant="champion"}} 60.0',
        f'{METRIC}{{le="+Inf",{REC},variant="challenger"}} 40.0',
        f'{METRIC}{{le="+Inf",route="/health",variant="none"}} 7.0',
    ]
)


def test_parse_histogram_sums_variants_and_ignores_other_routes() -> None:
    assert parse_histogram(BUCKETS) == {0.1: 0.0, 0.25: 100.0, float("inf"): 100.0}


def test_histogram_quantile_matches_prometheus_interpolation() -> None:
    # all 100 observations in (0.1, 0.25]: p95 = 0.1 + 0.15 * 0.95 (same as promtool test)
    after = {0.1: 0.0, 0.25: 100.0, float("inf"): 100.0}
    assert histogram_quantile(0.95, {}, after) == pytest.approx(0.2425)
    assert histogram_quantile(0.95, after, after) is None  # no new observations


def test_percentiles() -> None:
    stats = percentiles([float(i) for i in range(1, 101)])
    assert stats["p50_ms"] == pytest.approx(50.5)
    assert stats["max_ms"] == 100.0
    assert percentiles([]) == {}


def _reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


def test_read_response_content_length_and_chunked() -> None:
    async def scenario() -> None:
        fixed = b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\nhello"
        assert await read_response(_reader(fixed)) == (200, b"hello")
        chunked = (
            b"HTTP/1.1 503 Busy\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n"
        )
        assert await read_response(_reader(chunked)) == (503, b"abcde")

    asyncio.run(scenario())


def test_keep_alive_pool_reuses_connections() -> None:
    async def scenario() -> None:
        connections = 0

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal connections
            connections += 1
            while True:
                try:
                    request = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    break
                body = request.split(b" ", 2)[1]  # echo the path
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: %d\r\n\r\n%s" % (len(body), body))
                await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        pool = KeepAlivePool(f"http://127.0.0.1:{port}", size=2, timeout=5)
        await pool.open()
        results = await asyncio.gather(*(pool.get(f"/r/{i}") for i in range(20)))
        assert [body for _, body in results] == [f"/r/{i}".encode() for i in range(20)]
        assert connections == 2  # 20 requests over 2 persistent connections
        await pool.close()
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
