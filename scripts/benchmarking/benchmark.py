from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    operation: str,
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
            operation,
            {"error_type": type(error).__name__},
        )
        return 0
    tracker.observe((time.perf_counter() - started_at) * 1000.0, str(status), operation)
    return status


def _parse_headers(values: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"header must use Name: Value format: {value}")
        name, header_value = value.split(":", 1)
        headers[name.strip()] = header_value.strip()
    return headers


def run_benchmark(
    url: str,
    requests: int = 100,
    concurrency: int = 10,
    method: str = "GET",
    body: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if requests < 1:
        raise ValueError("requests must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    tracker = LatencyTracker()
    body_bytes = body.encode("utf-8") if body is not None else None
    request_headers = dict(headers or {})
    started_at = time.perf_counter()
    status_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _request_once,
                url,
                method.upper(),
                body_bytes,
                request_headers,
                timeout,
                tracker,
                "benchmark_request",
            )
            for _ in range(requests)
        ]
        for future in as_completed(futures):
            status = future.result()
            status_counts[str(status)] += 1
    elapsed = time.perf_counter() - started_at
    return {
        "url": url,
        "method": method.upper(),
        "requests": requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 6),
        "requests_per_second": round(requests / elapsed, 6) if elapsed else 0.0,
        "status_counts": dict(status_counts),
        "latency": tracker.summary(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a concurrent HTTP latency benchmark")
    parser.add_argument("--url", default="http://localhost:8000/healthz")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--method", default="GET")
    parser.add_argument("--body")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.url,
        args.requests,
        args.concurrency,
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
