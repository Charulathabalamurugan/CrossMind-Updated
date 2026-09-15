from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
)


def scan_secrets(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in {".git", ".venv", "__pycache__", "archive"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(str(path.relative_to(root)))
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CrossMind security checks")
    parser.add_argument("--skip-pip-audit", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    findings = scan_secrets(root)
    if findings:
        print("Potential secrets found:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1

    if not args.skip_pip_audit:
        try:
            subprocess.run([sys.executable, "-m", "pip_audit", "-f", "json"], check=False)
        except FileNotFoundError:
            print("pip-audit is not installed; skipping dependency audit", file=sys.stderr)
    print("Secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
