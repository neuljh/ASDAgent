import json
import os
import re
from pathlib import Path
from collections import OrderedDict
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("ASDAGENT_DATA_ROOT", REPO_ROOT / "data"))
PROFILE_ROOT = Path(os.getenv("ASDAGENT_PROFILE_ROOT", DATA_ROOT))


def load_metadata(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="gbk")
    df = df.set_index("index")
    return df


def extract_index(filename: str):
    match = re.match(r"(\d+)-", filename)
    if not match:
        return None
    return int(match.group(1))


def _to_native(val):
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return val

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
    txt_dir = DATA_ROOT / "processed" / "batch2"
    csv_path = PROFILE_ROOT / "data_info.csv"
    out_path = PROFILE_ROOT / "batch2_dialogue_info.jsonl"
    info_path = PROFILE_ROOT / "child_info.jsonl"

    hospital_by_name = load_hospital_by_name(info_path)
    meta = load_metadata(csv_path)

    records = []
    missing = []
    for path in sorted(txt_dir.glob("*.txt")):
        base_filename = path.name
        idx = base_filename.split("_")[0] if "_" in base_filename else '0'
        child_name = base_filename.split("_")[1] if "_" in base_filename else '小明'
        topic = base_filename.split("_")[-1] if "_" in base_filename else '日常交流'
        hospital_name = hospital_by_name.get(child_name)

        record = {
            "name": child_name,
            "topic": topic,
            "index": int(idx),
            "hospital": hospital_name,
            "file_name": base_filename,
        }
        records.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(records)} records to {out_path}")
    if missing:
        print(f"Warning: {len(missing)} files had no matching index: {missing}")


if __name__ == "__main__":
    # python -m asdagent.child_profile.process_data_batch2
    main()
