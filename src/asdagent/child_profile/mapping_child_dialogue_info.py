import json
import os
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("ASDAGENT_DATA_ROOT", REPO_ROOT / "data"))
PROFILE_ROOT = Path(os.getenv("ASDAGENT_PROFILE_ROOT", DATA_ROOT))


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_dialogues(paths: List[Path]) -> Dict[str, List[Dict]]:
    by_name: Dict[str, List[Dict]] = {}
    for p in paths:
        if not p.exists():
            continue
        for rec in _read_jsonl(p):
            name = rec.get("name")
            
            if not name:
                continue
            entry = {"topic": rec.get("topic"), "file_name": rec.get("file_name")}
            by_name.setdefault(name, []).append(entry)
    return by_name


def main():
    batch1_path = PROFILE_ROOT / "batch1_dialogue_info.jsonl"
    batch2_path = PROFILE_ROOT / "batch2_dialogue_info.jsonl"
    child_info_path = PROFILE_ROOT / "child_info.jsonl"
    out_path = PROFILE_ROOT / "mapping_child_dialogue_info.jsonl"

    dialogues_by_name = load_dialogues([batch1_path, batch2_path])

    outputs = []
    for rec in _read_jsonl(child_info_path):
        name = rec.get("姓名")
        file_infos = dialogues_by_name.get(name, [])
        if len(file_infos) == 0:
            nick_name = rec.get("小名")
            file_infos = dialogues_by_name.get(nick_name, [])
        combined = {
            "姓名": name,
            "小名": rec.get("小名"),
            "性别": rec.get("性别"),
            "医院": rec.get("医院"),
            "dialogue_number": len(file_infos),
            "file_infos": file_infos,
            "日期": rec.get("日期"),
            "生日": rec.get("生日"),
            "年龄": rec.get("年龄"),
            "语言生长发育年龄": rec.get("语言生长发育年龄"),
            "语言生日": rec.get("语言生日"),
            "现在的年龄": rec.get("现在的年龄"),
            "现在的语言生长发育年龄": rec.get("现在的语言生长发育年龄"),
        }
        outputs.append(combined)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in outputs:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(outputs)} records to {out_path}")


if __name__ == "__main__":
    # python -m asdagent.child_profile.mapping_child_dialogue_info
    main()
