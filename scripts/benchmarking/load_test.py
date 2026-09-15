from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarking.latency_tracker import LatencyTracker


def _request_once(
    url: str,
    method: str,
    body: Optional[bytes],
    headers: Dict[str, str],
    timeout: float,
    tracker: LatencyTracker,
) -> int:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = int(response.status)
    except urllib.error.HTTPError as error:
        error.read()
        status = int(error.code)
    except Exception as error:
        tracker.observe(
            (time.perf_counter() - started_at) * 1000.0,
            "error",
            "load_request",
            {"error_type": type(error).__name__},
        )
        return 0
    tracker.observe((time.perf_counter() - started_at) * 1000.0, str(status), "load_request")
    return status


def _parse_headers(values: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"header must use Name: Value format: {value}")
        name, header_value = value.split(":", 1)
        headers[name.strip()] = header_value.strip()
    return headers


def run_load_test(
    url: str,
    duration: float,
    concurrency: int,
    rate: Optional[float] = None,
    method: str = "GET",
    body: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if rate is not None and rate <= 0:
        raise ValueError("rate must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    tracker = LatencyTracker()
    status_counts: Counter[str] = Counter()
    body_bytes = body.encode("utf-8") if body is not None else None
    request_headers = dict(headers or {})
    delay = concurrency / rate if rate is not None else 0.0
    started_at = time.monotonic()
    deadline = started_at + duration

    def worker() -> None:
        next_request = started_at
        while time.monotonic() < deadline:
            status = _request_once(
                url,
                method.upper(),
                body_bytes,
                request_headers,
                timeout,
                tracker,
            )
            status_counts[str(status)] += 1
            if delay:
                next_request += delay
                time.sleep(max(0.0, next_request - time.monotonic()))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(lambda _: worker(), range(concurrency)))
    elapsed = time.monotonic() - started_at
    total = sum(status_counts.values())
    return {
        "url": url,
        "method": method.upper(),
        "duration_seconds": round(duration, 6),
        "elapsed_seconds": round(elapsed, 6),
        "concurrency": concurrency,
        "target_rate": rate,
        "requests_per_second": round(total / elapsed, 6) if elapsed else 0.0,
        "status_counts": dict(status_counts),
        "latency": tracker.summary(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a sustained HTTP load test")
    parser.add_argument("--url", default="http://localhost:8000/healthz")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--method", default="GET")
    parser.add_argument("--body")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_load_test(
        args.url,
        args.duration,
        args.concurrency,
        args.rate,
        args.method,
        args.body,
        _parse_headers(args.header),
        args.timeout,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
