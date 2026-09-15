from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def _normalize_helm(text: str) -> str:
    control = re.compile(r"{{-?\s*(if|else|else\s+if|end|range|with|define|block|template|include)\b.*?-?}}", re.DOTALL)
    text = control.sub("", text)
    return re.sub(r"{{-?.*?-?}}", "PLACEHOLDER", text)


def _validate_file(path: Path, root: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
        if path.relative_to(root).parts[:3] == ("helm", "crossmind", "templates"):
            text = _normalize_helm(text)
        else:
            text = re.sub(r"\{\{.*?\}\}", "PLACEHOLDER", text)
        list(yaml.safe_load_all(text))
    except Exception as exc:
        print(f"{path.relative_to(root)}: {exc}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CrossMind YAML manifests")
    parser.add_argument("paths", nargs="*", default=["monitoring", "kubernetes", ".github/workflows"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for value in args.paths:
        base = root / value
        for path in sorted(base.rglob("*.yml")) + sorted(base.rglob("*.yaml")):
            failures += _validate_file(path, root)
    if failures:
        return 1
    print("YAML validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
