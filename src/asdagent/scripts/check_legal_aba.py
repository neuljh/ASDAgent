#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("ASDAGENT_DATA_ROOT", REPO_ROOT / "data")) / "processed" / "all"
DOCTOR_TAGS = {"指令", "强化", "其他", "半辅助", "全辅助"}
CHILD_TAGS = {"相关的回答", "不相关的回答", "无响应", "重复"}
ROLE_PATTERN = re.compile(r"^(医生|儿童)[:：](.*)$")
TAG_PATTERN = re.compile(r"[（(]([^（）()]+)[）)]\s*$")


@dataclass(frozen=True)
class AbaIssue:
    file_name: str
    line_number: int
    role: str
    tag: str
    content: str


def extract_role_and_tag(line: str) -> tuple[str | None, str]:
    match = ROLE_PATTERN.match(line.strip())
    if not match:
        return None, ""

    role, rest = match.groups()
    tag_match = TAG_PATTERN.search(rest.strip())
    tag = tag_match.group(1).strip() if tag_match else ""
    return role, tag


def find_illegal_aba_tags(data_root: Path) -> list[AbaIssue]:
    issues: list[AbaIssue] = []

    for path in sorted(data_root.glob("*.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                role, tag = extract_role_and_tag(line)
                if role is None:
                    continue

                legal_tags = DOCTOR_TAGS if role == "医生" else CHILD_TAGS
                if tag not in legal_tags:
                    issues.append(
                        AbaIssue(
                            file_name=path.name,
                            line_number=line_number,
                            role=role,
                            tag=tag,
                            content=line,
                        )
                    )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find doctor/child ABA tags outside the allowed label sets.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Directory containing processed dialogue txt files. Default: {DATA_ROOT}",
    )
    args = parser.parse_args()

    issues = find_illegal_aba_tags(args.data_root)
    print(f"Scanned directory: {args.data_root}")
    print(f"Illegal ABA tag lines: {len(issues)}")

    for issue in issues:
        shown_tag = issue.tag or "<missing>"
        print(
            f"{issue.file_name}:{issue.line_number} | {issue.role} | "
            f"tag={shown_tag} | {issue.content}"
        )


if __name__ == "__main__":
    # python -m asdagent.scripts.check_legal_aba --data-root data/processed/all
    main()
