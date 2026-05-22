import json
import os
import re
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("ASDAGENT_DATA_ROOT", REPO_ROOT / "data"))
PROFILE_ROOT = Path(os.getenv("ASDAGENT_PROFILE_ROOT", DATA_ROOT))


def parse_filename(filename: str):
    """
    Accepts patterns like:
    111_林昴成_20250115_过年活动.txt
    240_董涵钰_2025.03.02_兔子.txt
    258_小明_12.19_旅游.txt
    """
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    parts = stem.split("_", 3)
    if len(parts) < 4:
        return None
    idx_str, name, _, topic = parts
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    return idx, name, topic


def load_hospital_by_name(info_path: Path):
    mapping = OrderedDict()  # preserve first occurrence
    if not info_path.exists():
        return mapping
    with info_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("姓名")
            hospital = obj.get("医院")
            if name and hospital and name not in mapping:
                mapping[name] = hospital
    return mapping


def main():
    txt_dir = DATA_ROOT / "processed" / "batch1"
    out_path = PROFILE_ROOT / "batch1_dialogue_info.jsonl"
    info_path = PROFILE_ROOT / "child_info.jsonl"

    hospital_by_name = load_hospital_by_name(info_path)

    records = []
    for path in sorted(txt_dir.glob("*.txt")):
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        idx, name, topic = parsed
        record = {
            "name": name,
            "topic": topic,
            "index": idx,
            "hospital": hospital_by_name.get(name),
            "file_name": path.name,
        }
        records.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    # python -m asdagent.child_profile.process_data_batch1
    main()
