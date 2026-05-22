import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = Path(os.getenv("ASDAGENT_PROFILE_ROOT", REPO_ROOT / "data"))
SRC = PROFILE_ROOT / "child_profiles.generated.jsonl"
DST = PROFILE_ROOT / "child_profiles.filtered.jsonl"


def main() -> None:
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return

    kept = 0
    dst_dir = DST.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    with SRC.open("r", encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("current_dialogue_number", 0) == 0:
                continue
            json.dump(rec, fout, ensure_ascii=False)
            fout.write("\n")
            kept += 1
    print(f"Kept {kept} records -> {DST}")


if __name__ == "__main__":
    # python -m asdagent.child_profile.remove_zero_child
    main()
