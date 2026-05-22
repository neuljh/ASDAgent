import json
import os
import re
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = os.path.join(os.getenv("ASDAGENT_DATA_ROOT", os.path.join(REPO_ROOT, "data")), "processed", "all")
PROFILE_ROOT = os.getenv("ASDAGENT_PROFILE_ROOT", os.path.join(REPO_ROOT, "data"))
MAPPING_PATH = os.path.join(PROFILE_ROOT, "mapping_child_dialogue_info.jsonl")
OUTPUT_PATH = os.path.join(PROFILE_ROOT, "child_profiles.generated.jsonl")
OUT_DIR = os.path.join(os.getenv("ASDAGENT_OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs")), "figures")

STRATEGY_ALIASES = {
    "其他": "其他",
    "半辅助": "半辅助",
    "全辅助": "全辅助",
    "强化": "强化",
    "指令": "指令",
}

CHILD_TYPES = {"不相关的回答", "相关的回答", "重复", "无响应"}
NON_INSTRUCTION_STRATEGIES = {"其他", "强化"}


def _age_to_years(age_str: str) -> str:
    """Convert age formats like '4Y6M' to '4.5岁'. If parse fails, return original."""
    if not isinstance(age_str, str):
        return str(age_str)
    match = re.match(r"(?i)(\d+)\s*[Y年岁]?\s*(\d+)?\s*M?", age_str.replace(" ", ""))
    if match:
        years = int(match.group(1))
        months = int(match.group(2) or 0)
        value = years + months / 12.0
        return f"{value:.1f}岁"
    return age_str


def _age_to_float(age_str: str) -> float:
    """Return age in years as float; fallback 0.0 on failure."""
    norm = _age_to_years(age_str)
    try:
        if isinstance(norm, str) and norm.endswith("岁"):
            return float(norm[:-1])
        return float(norm)
    except Exception:
        return 0.0


def _parse_line(line: str) -> Tuple[str, str, str]:
    """Return role, tag (strategy or type), content."""
    text = line.strip()
    role = "未知"
    tag = ""
    # normalize punctuation for parsing
    text_norm = text.replace(":", "：")
    if "：" in text_norm:
        role, rest = text_norm.split("：", 1)
        content = rest.strip()
    else:
        content = text_norm
    # extract tag inside the last parentheses (Chinese or ASCII)
    m = re.search(r"[（(]([^（）()]+)[）)]\s*$", content)
    if m:
        tag = m.group(1).strip()
        content = content[: m.start()].rstrip()
    role = role.strip()
    return role, tag, content


def _normalize_strategy(tag: str) -> str:
    return STRATEGY_ALIASES.get(tag, tag or "未知")


def _normalize_child_type(tag: str) -> str:
    return tag if tag in CHILD_TYPES else "未知"


def _parse_dialogue_file(path: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            role, tag, content = _parse_line(line)
            if role.startswith("医"):
                entries.append({"role": "医生", "strategy": _normalize_strategy(tag), "content": content})
            elif role.startswith("儿"):
                entries.append({"role": "儿童", "type": _normalize_child_type(tag), "content": content})
    return entries


def _count_turns(lines: List[Dict[str, str]]) -> int:
    return max(1, sum(1 for e in lines if e["role"] == "儿童"))


def _stats_for_entries(entries: List[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return counts/probs under last_strategy and sequential routes."""
    # Initialize
    strategies = set(STRATEGY_ALIASES.values())
    # strategies.update(["辅助"])  # ensure base
    last_counts = {s: defaultdict(int) for s in strategies}
    seq_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    pending_doctor: List[str] = []
    for ent in entries:
        if ent["role"] == "医生":
            pending_doctor.append(ent.get("strategy", "未知"))
        else:
            ctype = ent.get("type", "未知")
            if ctype == "未知":
                print()
            if pending_doctor:
                # last strategy route
                last_counts[pending_doctor[-1]][ctype] += 1
                # sequential route with full strategy path
                seq_key = "，".join(pending_doctor)
                seq_counts[seq_key][ctype] += 1
            pending_doctor = []

    def _to_probs(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        probs: Dict[str, Dict[str, float]] = {}
        for strat, ct in counts.items():
            total = sum(ct.values())
            probs[strat] = {k: (v / total if total else 0.0) for k, v in ct.items()}
        return probs

    return {
        "last_counts": last_counts,
        "last_probs": _to_probs(last_counts),
        "sequential_counts": seq_counts,
        "sequential_probs": _to_probs(seq_counts),
    }


def _bigram_non_instruction(entries: List[Dict[str, str]]) -> Tuple[int, int]:
    """Return (child_after, doctor_after) counts for non-instruction strategies."""
    child_after = 0
    doctor_after = 0
    for i, ent in enumerate(entries):
        if ent["role"] == "医生" and ent.get("strategy") in NON_INSTRUCTION_STRATEGIES:
            if i + 1 < len(entries):
                nxt = entries[i + 1]
                if nxt["role"] == "儿童":
                    child_after += 1
                elif nxt["role"] == "医生":
                    doctor_after += 1
    return child_after, doctor_after


def _aggregate_counts(count_dicts: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    agg: Dict[str, Dict[str, int]] = {}
    for cd in count_dicts:
        for strat, m in cd.items():
            if strat not in agg:
                agg[strat] = defaultdict(int)
            for k, v in m.items():
                agg[strat][k] += v
    return agg


def load_child_entry_by_filename(filename: str) -> Dict:
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for info in entry.get("file_infos", []):
                if info.get("file_name") == filename:
                    return entry
    return {}


def compute_child_profile(filename: str) -> Dict[str, str]:
    entry = load_child_entry_by_filename(filename)
    profile = {
        "name": entry.get("姓名", "未知"),
        "nickname": entry.get("小名", ""),
        "gender": entry.get("性别", "未知"),
        "age": _age_to_years(entry.get("年龄", entry.get("现在的年龄", "未知"))),
        "verbal_level": _age_to_years(entry.get("语言生长发育年龄", "未知")),
    }
    return profile


def _get_child_files(entry: Dict) -> List[str]:
    files = []
    for info in entry.get("file_infos", []):
        if "file_name" in info:
            files.append(info["file_name"])
    return files


def compute_gender_age_stats(mapping_path: str = MAPPING_PATH) -> Dict[str, Any]:
    total = 0
    male = 0
    female = 0
    ages: List[float] = []
    lang_ages: List[float] = []
    ages_by_gender: Dict[str, List[float]] = {"男": [], "女": []}
    lang_by_gender: Dict[str, List[float]] = {"男": [], "女": []}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            gender = entry.get("性别") or entry.get("gender")
            if gender == "男":
                male += 1
            elif gender == "女":
                female += 1
            age_raw = entry.get("年龄") or entry.get("现在的年龄", None)
            lang_raw = entry.get("语言生长发育年龄") or entry.get("现在的语言生长发育年龄", None)
            if age_raw not in (None, "", "null"):
                val = _age_to_float(age_raw)
                ages.append(val)
                if gender in ages_by_gender:
                    ages_by_gender[gender].append(val)
            if lang_raw not in (None, "", "null"):
                val = _age_to_float(lang_raw)
                lang_ages.append(val)
                if gender in lang_by_gender:
                    lang_by_gender[gender].append(val)

    def _mean_std(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        return {"mean": mean, "std": std}

    return {
        "total": total,
        "male": male,
        "female": female,
        "age": _mean_std(ages),
        "language_age": _mean_std(lang_ages),
        "age_by_gender": {g: _mean_std(vals) for g, vals in ages_by_gender.items()},
        "language_age_by_gender": {g: _mean_std(vals) for g, vals in lang_by_gender.items()},
    }


def compute_stats_for_file(filename: str) -> Dict[str, any]:
    path = os.path.join(DATA_ROOT, filename)
    entries = _parse_dialogue_file(path)
    turns = _count_turns(entries)
    stats = _stats_for_entries(entries)
    child_after, doctor_after = _bigram_non_instruction(entries)
    bigram_total = child_after + doctor_after
    bigram_prob = {
        "child_after": child_after,
        "doctor_after": doctor_after,
        "child_after_prob": child_after / bigram_total if bigram_total else 0.0,
        "doctor_after_prob": doctor_after / bigram_total if bigram_total else 0.0,
    }
    return {
        "turns": turns,
        "stats": stats,
        "bigram": bigram_prob,
    }


def compute_stats_for_files(filenames: List[str]) -> Dict[str, any]:
    all_entries: List[Dict[str, str]] = []
    count_last = []
    count_seq = []
    child_after_total = 0
    doctor_after_total = 0
    turns_list: List[int] = []
    doctor_lengths_overall: List[int] = []
    child_lengths_overall: List[int] = []
    doctor_lengths_by_strategy: Dict[str, List[int]] = defaultdict(list)
    child_lengths_by_type: Dict[str, List[int]] = defaultdict(list)
    for fname in filenames:
        print(f'file: {fname}')
        path = os.path.join(DATA_ROOT, fname)
        if not os.path.exists(path):
            continue
        entries = _parse_dialogue_file(path)
        all_entries.extend(entries)
        s = _stats_for_entries(entries)
        count_last.append(s["last_counts"])
        count_seq.append(s["sequential_counts"])
        ca, da = _bigram_non_instruction(entries)
        child_after_total += ca
        doctor_after_total += da
        turns_list.append(_count_turns(entries))
        for ent in entries:
            if ent["role"] == "医生":
                length = len(ent.get("content", ""))
                doctor_lengths_overall.append(length)
                doctor_lengths_by_strategy[ent.get("strategy", "未知")].append(length)
            elif ent["role"] == "儿童":
                length = len(ent.get("content", "")) if ent.get("type") != "无响应" else 0
                child_lengths_overall.append(length)
                child_lengths_by_type[ent.get("type", "未知")].append(length)

    agg_last = _aggregate_counts(count_last)
    agg_seq = _aggregate_counts(count_seq)

    def _to_probs(agg: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        probs: Dict[str, Dict[str, float]] = {}
        for strat, m in agg.items():
            total = sum(m.values())
            probs[strat] = {k: (v / total if total else 0.0) for k, v in m.items()}
        return probs

    bigram_total = child_after_total + doctor_after_total
    bigram_prob = {
        "child_after": child_after_total,
        "doctor_after": doctor_after_total,
        "child_after_prob": child_after_total / bigram_total if bigram_total else 0.0,
        "doctor_after_prob": doctor_after_total / bigram_total if bigram_total else 0.0,
    }

    def _mean_std(vals: List[int]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        return {"mean": mean, "std": std}

    return {
        "stats": {
            "last_counts": agg_last,
            "last_probs": _to_probs(agg_last),
            "sequential_counts": agg_seq,
            "sequential_probs": _to_probs(agg_seq),
        },
        "bigram": bigram_prob,
        "turns": _mean_std(turns_list),
        "utterance_lengths": {
            "doctor_overall": _mean_std(doctor_lengths_overall),
            "child_overall": _mean_std(child_lengths_overall),
            "doctor_by_strategy": {k: _mean_std(v) for k, v in doctor_lengths_by_strategy.items()},
            "child_by_type": {k: _mean_std(v) for k, v in child_lengths_by_type.items()},
        },
        "turns_list": turns_list,
    }


def _to_plain(obj: Any) -> Any:
    """Recursively convert defaultdicts to dicts for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def compute_current_child_info(
    mapping_path: str = MAPPING_PATH,
    data_root: str = DATA_ROOT,
    output_path: str = OUTPUT_PATH,
) -> None:
    records: List[Dict[str, Any]] = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            files = _get_child_files(entry)
            existing_files = [fn for fn in files if os.path.exists(os.path.join(data_root, fn))]
            current_dialogue_number = len(existing_files)

            record: Dict[str, Any] = {
                "姓名": entry.get("姓名"),
                "小名": entry.get("小名"),
                "性别": entry.get("性别"),
                "年龄": entry.get("年龄"),
                "语言生长发育年龄": entry.get("语言生长发育年龄"),
                "现在的年龄": entry.get("现在的年龄"),
                "现在的语言生长发育年龄": entry.get("现在的语言生长发育年龄"),
                "file_infos": entry.get("file_infos", []),
                "current_dialogue_number": current_dialogue_number,
            }

            if current_dialogue_number == 0:
                record.update(
                    {
                        "last_probs": None,
                        "sequential_probs": None,
                        "bigram": None,
                    }
                )
            else:
                agg = compute_stats_for_files(existing_files)
                record.update(
                    {
                        "last_probs": _to_plain(agg.get("stats", {}).get("last_probs")),
                        "sequential_probs": _to_plain(agg.get("stats", {}).get("sequential_probs")),
                        "bigram": _to_plain(agg.get("bigram")),
                    }
                )
            records.append(record)

    out_dir = os.path.dirname(output_path)
    # if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            json.dump(rec, out_f, ensure_ascii=False)
            out_f.write("\n")
    print(f"Saved {len(records)} records to {output_path}")


def test() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_file = "1_demo_child_20260101_sample_topic.txt"
    print("=== Child profile ===")
    print(json.dumps(compute_child_profile(sample_file), ensure_ascii=False, indent=2))

    print("\n=== Per-file stats ===")
    print(json.dumps(compute_stats_for_file(sample_file), ensure_ascii=False, indent=2))

    entry = load_child_entry_by_filename(sample_file)
    child_files = _get_child_files(entry)
    print("\n=== Child-aggregated stats ===")
    print(json.dumps(compute_stats_for_files(child_files), ensure_ascii=False, indent=2))

    # Global across all files
    all_files = [fn for fn in os.listdir(DATA_ROOT) if fn.endswith(".txt")]
    global_stats = compute_stats_for_files(all_files)
    print("\n=== Global stats ===")
    print(json.dumps(global_stats, ensure_ascii=False, indent=2))

    demo_stats = compute_gender_age_stats()
    print("\n=== Demographics ===")
    print(json.dumps(demo_stats, ensure_ascii=False, indent=2))

    # Persist global stats
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(os.path.dirname(OUTPUT_PATH), "child_global_stats.json")
    payload = {"global_stats": global_stats, "demographics": demo_stats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved global stats to {out_path}")

    # Plot turn distribution
    try:
        turns = global_stats.get("turns_list", [])
        if turns:
            freq: Dict[int, int] = defaultdict(int)
            for t in turns:
                freq[int(t)] += 1
            xs = sorted(freq.keys())
            ys = [freq[x] for x in xs]
            plt.figure(figsize=(8, 4))
            plt.bar(xs, ys, width=0.6, color="#4C72B0")
            plt.xlabel("Turns")
            plt.ylabel("Count")
            plt.title("Turn Count Frequency")
            plt.grid(axis="y", linestyle="--", alpha=0.4)
            img_path = os.path.join(out_dir, "turn_distribution.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close()
            print(f"Saved turn frequency plot to {img_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    # python -m asdagent.child_profile.compute_current_child_info
    compute_current_child_info()
