"""Async HTTP load generator for ``GET /recommendations/{user_id}``.

Two modes:

* **closed loop** (``--concurrency N``): N workers send back-to-back requests for
  ``--duration`` seconds -> maximum sustainable throughput.
* **open loop** (``--rate R``): requests are *scheduled* at a fixed rate regardless of
  how fast responses come back, and latency is measured from the scheduled start. This
  includes queueing delay and avoids coordinated omission (a closed-loop test under-reports
  tail latency when the server stalls).

User mix: ``--cold-fraction`` of requests use ids unknown to the model (cold-start path);
the rest are sampled uniformly from the model package's known users.

The report contains client-side percentiles and the server-side percentiles computed from
the delta of the API's Prometheus histogram over the run (same method as
``histogram_quantile``), plus the hardware context. HTTP/1.1 is spoken directly over
asyncio streams with a keep-alive pool (see ``KeepAlivePool`` for why not httpx).

    python -m recsys.loadtest --rate 300 --duration 30 --label open-300rps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from recsys.utils import hardware_info

ROUTE = "/recommendations/{user_id}"
_BUCKET_PREFIX = "recsys_http_request_duration_seconds_bucket{"
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def known_users(package_dir: Path) -> np.ndarray:
    with np.load(Path(package_dir) / "model" / "arrays.npz") as npz:
        return npz["user_ids"].copy()


def percentiles(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    arr = np.asarray(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(arr.max()), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def parse_histogram(text: str) -> dict[float, float]:
    """Cumulative bucket counts of the recommendations route, summed over variants."""
    buckets: dict[float, float] = {}
    for line in text.splitlines():
        if not line.startswith(_BUCKET_PREFIX):
            continue
        label_part, _, value = line.rpartition("} ")
        labels = dict(_LABEL_RE.findall(label_part))
        if labels.get("route") != ROUTE:
            continue
        le = labels["le"]
        key = float("inf") if le == "+Inf" else float(le)
        buckets[key] = buckets.get(key, 0.0) + float(value)
    return buckets


def histogram_quantile(
    q: float, before: dict[float, float], after: dict[float, float]
) -> float | None:
    """Prometheus-style quantile (linear interpolation inside the bucket) of a histogram delta."""
    bounds = sorted(after)
    counts = [after[b] - before.get(b, 0.0) for b in bounds]
    total = counts[-1] if counts else 0.0
    if total <= 0:
        return None
    rank = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound, count in zip(bounds, counts, strict=True):
        if count >= rank:
            if bound == float("inf"):
                return prev_bound
            width = count - prev_count
            return prev_bound + (bound - prev_bound) * ((rank - prev_count) / width if width else 0)
        prev_bound, prev_count = bound, count
    return None


class Recorder:
    def __init__(self) -> None:
        self.latencies: list[float] = []  # from the scheduled send time (open loop)
        self.service_latencies: list[float] = []  # from the actual send time
        self.dispatch_lag: list[float] = []  # actual send - scheduled send
        self.status: Counter[str] = Counter()
        self.strategies: Counter[str] = Counter()
        self.variants: Counter[str] = Counter()
        self.cached = 0

    def record(
        self, scheduled: float, sent: float, status: int | None, body: bytes, error: str = ""
    ) -> None:
        done = time.perf_counter()
        self.latencies.append((done - scheduled) * 1000)
        self.service_latencies.append((done - sent) * 1000)
        self.dispatch_lag.append((sent - scheduled) * 1000)
        if status is None:
            self.status[f"error:{error}"] += 1
            return
        self.status[str(status)] += 1
        if status == 200:
            payload = json.loads(body)
            self.strategies[payload["strategy"]] += 1
            self.variants[payload["variant"]] += 1
            self.cached += int(payload["cached"])


async def read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one HTTP/1.1 response (Content-Length or chunked body)."""
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if "content-length" in headers:
        return status, await reader.readexactly(int(headers["content-length"]))
    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks = []
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";")[0].strip(), 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                return status, b"".join(chunks)
            chunks.append(await reader.readexactly(size))
            await reader.readexactly(2)
    return status, b""


class KeepAlivePool:
    """Minimal HTTP/1.1 keep-alive connection pool on raw asyncio streams.

    On this Windows machine httpx's AsyncClient was itself the bottleneck (~70 req/s at 32
    concurrent requests, p50 ~300 ms) while this pool pushed ~2.5k req/s against the same
    server - so the load generator speaks HTTP/1.1 directly.
    """

    def __init__(self, url: str, size: int, timeout: float) -> None:
        parsed = urllib.parse.urlsplit(url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.size = size
        self.timeout = timeout
        self._idle: asyncio.Queue = asyncio.Queue()
        self._replacements: set[asyncio.Task] = set()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_connection(self.host, self.port)

    async def open(self) -> None:
        for _ in range(self.size):
            self._idle.put_nowait(await self._connect())

    async def _replace(self) -> None:
        self._idle.put_nowait(await self._connect())

    async def get(self, path: str) -> tuple[int, bytes]:
        reader, writer = await self._idle.get()
        try:
            writer.write(f"GET {path} HTTP/1.1\r\nHost: {self.host}\r\n\r\n".encode())
            await writer.drain()
            result = await asyncio.wait_for(read_response(reader), self.timeout)
        except BaseException:
            writer.close()  # broken connection: replace it so the pool keeps its size
            task = asyncio.create_task(self._replace())
            self._replacements.add(task)
            task.add_done_callback(self._replacements.discard)
            raise
        self._idle.put_nowait((reader, writer))
        return result

    async def close(self) -> None:
        while not self._idle.empty():
            _, writer = self._idle.get_nowait()
            writer.close()


async def _one(pool: KeepAlivePool, user: int, k: int, scheduled: float, rec: Recorder | None):
    sent = time.perf_counter()
    try:
        status, body = await pool.get(f"/recommendations/{user}?k={k}")
    except (OSError, TimeoutError, asyncio.IncompleteReadError, ValueError) as exc:
        if rec is not None:
            rec.record(scheduled, sent, None, b"", type(exc).__name__)
        return
    if rec is not None:
        rec.record(scheduled, sent, status, body)


async def _wait_until(deadline: float) -> None:
    """Sleep until ``deadline``; spin the last 2 ms (Windows timers are ~1-16 ms coarse)."""
    delay = deadline - time.perf_counter()
    if delay > 0.002:
        await asyncio.sleep(delay - 0.002)
    while time.perf_counter() < deadline:
        await asyncio.sleep(0)


def _user_stream(users: np.ndarray, cold_fraction: float, seed: int):
    rng = random.Random(seed)
    cold_base = int(users.max()) + 1_000_000
    while True:
        if rng.random() < cold_fraction:
            yield cold_base + rng.randrange(1_000_000)
        else:
            yield int(users[rng.randrange(len(users))])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    users = known_users(Path(args.package))
    if args.users_from_events:  # users with streamed events -> exercises the real-time path
        from recsys.streaming.sinks import read_events

        users = read_events(Path(args.users_from_events))["user_id"].unique().astype(np.int64)
    stream = _user_stream(users, args.cold_fraction, args.seed)
    size = args.connections if args.rate else max(args.connections, args.concurrency)
    pool = KeepAlivePool(args.url, size, args.timeout)
    await pool.open()
    try:
        for _ in range(args.warmup):  # warm caches/allocators; not recorded
            await _one(pool, next(stream), args.k, time.perf_counter(), None)
        before = parse_histogram((await pool.get("/metrics"))[1].decode())

        rec = Recorder()
        started = time.perf_counter()
        if args.rate:
            interval = 1.0 / args.rate
            pending = []
            for i in range(int(args.rate * args.duration)):
                scheduled = started + i * interval
                await _wait_until(scheduled)
                pending.append(
                    asyncio.create_task(_one(pool, next(stream), args.k, scheduled, rec))
                )
            await asyncio.gather(*pending)
        else:
            deadline = started + args.duration

            async def worker() -> None:
                while time.perf_counter() < deadline:
                    await _one(pool, next(stream), args.k, time.perf_counter(), rec)

            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started
        after = parse_histogram((await pool.get("/metrics"))[1].decode())
        ready = json.loads((await pool.get("/ready"))[1])
    finally:
        await pool.close()

    ok = rec.status.get("200", 0)
    server = {
        f"p{int(q * 100)}_ms": (
            round(v * 1000, 3) if (v := histogram_quantile(q, before, after)) is not None else None
        )
        for q in (0.5, 0.95, 0.99)
    }
    return {
        "label": args.label,
        "mode": "open-loop" if args.rate else "closed-loop",
        "config": {
            "url": args.url,
            "rate_rps": args.rate,
            "concurrency": None if args.rate else args.concurrency,
            "duration_s": args.duration,
            "k": args.k,
            "cold_fraction": args.cold_fraction,
            "sampled_users": len(users),
            "users_from_events": args.users_from_events,
            "warmup_requests": args.warmup,
        },
        "requests": len(rec.latencies),
        "ok": ok,
        "errors": len(rec.latencies) - ok,
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(len(rec.latencies) / elapsed, 1),
        "client_latency": percentiles(rec.latencies),
        "client_latency_from_send": percentiles(rec.service_latencies),
        "dispatch_lag": percentiles(rec.dispatch_lag),
        "server_latency_from_histogram": server,
        "status_codes": dict(rec.status),
        "strategies": dict(rec.strategies),
        "variants": dict(rec.variants),
        "cache_hit_ratio": round(rec.cached / ok, 4) if ok else None,
        "server": {
            "cache_backend": ready.get("cache_backend"),
            "models": [
                (m["variant"], m["model_type"], m["version"], m["n_items"]) for m in ready["models"]
            ],
        },
        "hardware": hardware_info(),
        "note": "load generator and API ran on the same machine (shared CPU)",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recsys loadtest", description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--package", default="artifacts/models/champion", help="model package (user ids)"
    )
    p.add_argument("--rate", type=float, default=None, help="open-loop requests/second")
    p.add_argument("--concurrency", type=int, default=32, help="closed-loop workers")
    p.add_argument("--connections", type=int, default=64)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--cold-fraction", type=float, default=0.1)
    p.add_argument(
        "--users-from-events", default=None, help="sample users from a streamed-events dir"
    )
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="loadtest")
    p.add_argument("--out", default="reports/loadtest")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = asyncio.run(run(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "label",
                    "mode",
                    "throughput_rps",
                    "client_latency",
                    "server_latency_from_histogram",
                    "errors",
                    "cache_hit_ratio",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    main()
