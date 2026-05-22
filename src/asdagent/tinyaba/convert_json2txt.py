#!/usr/bin/env python3
"""Convert dialogue_history in jsonl files to txt files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path(os.getenv("ASDAGENT_OUTPUT_ROOT", REPO_ROOT / "outputs"))


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def process_file(src: Path, src_root: Path, dst_root: Path) -> None:
    rel = src.relative_to(src_root)
    dst = dst_root / rel.with_suffix(".txt")
    dst.parent.mkdir(parents=True, exist_ok=True)

    lines_out = []
    for obj in load_jsonl(src):
        mem = obj.get("memory", {})
        history = mem.get("dialogue_history", [])
        if not isinstance(history, list):
            continue
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            tag = turn.get("strategy") or turn.get("type") or turn.get("tag") or ""
            if not content:
                continue
            suffix = f"（{tag}）" if tag else ""
            lines_out.append(f"{role}: {content}{suffix}")

    dst.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"Saved {dst}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert generated JSONL dialogue sessions to TXT files.")
    parser.add_argument("--model-name", default=os.getenv("ASDAGENT_MODEL", "gpt-4o-mini"))
    parser.add_argument("--src-root", type=Path, default=None, help="Directory containing generated JSONL files.")
    parser.add_argument("--dst-root", type=Path, default=None, help="Directory for converted TXT files.")
    args = parser.parse_args()

    src_root = args.src_root or DEFAULT_OUTPUT_ROOT / "json_files" / args.model_name
    dst_root = args.dst_root or DEFAULT_OUTPUT_ROOT / "txt_files" / args.model_name

    for src in src_root.rglob("*.jsonl"):
        process_file(src, src_root=src_root, dst_root=dst_root)


if __name__ == "__main__":
    # python -m asdagent.tinyaba.convert_json2txt
    main()
