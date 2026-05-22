# ASDAgent

[![Venue](https://img.shields.io/badge/ACL-2026%20Main-2f6fdd)](https://2026.aclweb.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Package](https://img.shields.io/badge/Package-src%2Fasdagent-orange)](src/asdagent)

Official implementation for **From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset**.

ASDAgent is a strategy-aware dual-agent framework for generating and analyzing autism intervention dialogues. It uses a **DoctorAgent** to perform ABA-aligned intervention planning and a **ChildAgent** to simulate heterogeneous child responses. The generated dialogues can be used for synthesis, analysis, and downstream distillation into smaller local models.

> **Clinical data notice**  
> Real clinical dialogue transcripts, identifiable child profiles, and derived private statistics are not included in this public release. The files under `data/*.sample.*` are synthetic placeholders for smoke testing and format reference only.

## Highlights

- **O-T-A-C reasoning loop**: `DoctorAgent` follows Observe, Think, Act, and Correct stages for strategy-aware intervention.
- **ABA strategy control**: doctor turns are constrained to atomic labels such as `指令`, `强化`, `半辅助`, `全辅助`, and `其他`.
- **Probabilistic ChildAgent**: child response types are sampled from personal/global behavior statistics when available.
- **Correct module**: generated doctor utterances can be segmented and filtered so the final response better matches the intended ABA strategy.
- **Open-source safe layout**: this release removes local paths, `.env` files, API logs, caches, and private clinical data.

## Repository Layout

```text
ASDAgent/
├── src/asdagent/
│   ├── tinyaba/                 # Dual-agent dialogue synthesis
│   │   ├── main.py              # DoctorAgent, ChildAgent, World, CLI
│   │   ├── prompts/             # Prompt templates for O-T-A-C stages
│   │   ├── json_schema/         # Pydantic schemas for structured outputs
│   │   ├── convert_json2txt.py  # Convert generated JSONL sessions to TXT
│   │   └── rename_syn_files.py  # Optional synthetic-file export helper
│   ├── child_profile/           # Profile/statistics preprocessing utilities
│   ├── scripts/                 # Validation scripts
│   ├── evaluation/              # Figure and analysis helpers
│   └── utils/                   # OpenAI-compatible API wrappers
├── data/                        # Synthetic sample inputs only
├── outputs/                     # Generated files, ignored by git
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd ASDAgent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For full LLM-backed synthesis, configure an OpenAI-compatible endpoint:

```bash
cp .env.example .env
# edit .env and set ASDAGENT_API_KEY
```

The code reads these variables:

```bash
ASDAGENT_MODEL=gpt-4o-mini
ASDAGENT_BASE_URL=https://api.openai.com/v1
ASDAGENT_API_KEY=...
ASDAGENT_DATA_ROOT=./data
ASDAGENT_OUTPUT_ROOT=./outputs
```

## Quick Start

Run a short synthesis job with the bundled synthetic sample profile:

```bash
python -m asdagent.tinyaba.main \
  --topics-path data/topics.sample.json \
  --profile-path data/child_profiles.sample.jsonl \
  --child-type-path data/child_type_map.sample.json \
  --global-stats-path data/child_global_stats.sample.json \
  --turns 4 \
  --output-dir outputs/json_files
```

Without `ASDAGENT_API_KEY`, the package returns schema-valid mock outputs for smoke testing. Set a real API key for meaningful dialogue synthesis.

Convert generated JSONL sessions to readable TXT:

```bash
python -m asdagent.tinyaba.convert_json2txt \
  --model-name gpt-4o-mini
```

Validate ABA labels in processed TXT dialogues:

```bash
python -m asdagent.scripts.check_legal_aba \
  --data-root data/processed/all
```

## Data Format

Processed dialogue files use one utterance per line:

```text
医生: 我们来看这张图片。你看到了什么？（指令）
儿童: 小狗。（相关的回答）
医生: 很棒，你说对了，是小狗。（强化）
医生: 小狗旁边还有什么？（指令）
儿童: 我想玩积木。（不相关的回答）
医生: 我听到你想玩积木。（其他）
医生: 我们先看图片，小狗旁边是一个红色的球。你可以说“球”。（半辅助）
儿童: [儿童无响应]（无响应）
医生: 球。（全辅助）
儿童: 球。（重复）
```

Child profile statistics are JSONL records. The sample file shows the expected keys:

```json
{
  "小名": "小明",
  "年龄": "5Y0M",
  "语言生长发育年龄": "3Y6M",
  "last_probs": {
    "指令": {
      "相关的回答": 0.35,
      "不相关的回答": 0.35,
      "重复": 0.10,
      "无响应": 0.20
    }
  },
  "bigram": {
    "child_after_prob": 0.45,
    "doctor_after_prob": 0.55
  }
}
```

Before using private or clinical data, de-identify all names, dates, locations, raw transcripts, and derived metadata according to your institutional review and data-use requirements.

## Core Components

**DoctorAgent**  
The doctor agent observes the latest child response, selects an intervention strategy, generates the next utterance, and optionally corrects strategy drift through segment-level filtering.

**ChildAgent**  
The child agent samples response types from personal and global statistics, then generates role-consistent child utterances through response-type-specific prompts.

**World**  
The world object coordinates turn-taking, memory, interruptions, persistence, and output paths.

## Citation

If this repository is useful for your research, please cite the following papers:

```bibtex
@article{lai2026synthesis,
  title={From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset},
  author={Lai, Junhong and Lai, Shuzhong and Yu, Yanhao and Chen, Wanlin and Yan, Chenyu and Li, Haifeng and Yao, Lin and Wang, Yueming},
  journal={arXiv preprint arXiv:2605.02916},
  year={2026}
}
```
```bibtex
@inproceedings{lai2025asd,
  title={ASD-iLLM: An Intervention Large Language Model for Autistic Children based on Real Clinical Dialogue Intervention Dataset},
  author={Lai, Shuzhong and Li, Chenxi and Lai, Junhong and Zhong, Yucun and Yan, Chenyu and Li, Xiang and Li, Haifeng and Pan, Gang and Yao, Lin and Wang, Yueming},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages={8058--8079},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE). The public code release does not include the private clinical dataset.
