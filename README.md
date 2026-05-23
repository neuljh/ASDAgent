# 🧩 [ACL 2026 Main] From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset

[![Venue: ACL 2026](https://img.shields.io/badge/Venue-ACL%202026%20Main-blue.svg)](https://2026.aclweb.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ASDAgent--Dataset-yellow)](https://huggingface.co/datasets/neuljh/ASDAgent-Dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


This is the official repository for the ACL 2026 Main Conference paper: **"[From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset]"**.

# 🤖 ASDAgent

<img width="7103" height="2978" alt="image" src="https://github.com/user-attachments/assets/c6c7ec5d-374e-4eff-8c09-f215f0e4f473" />


ASDAgent is a strategy-aware dual-agent framework for generating and analyzing autism intervention dialogues. It uses a **DoctorAgent** to perform ABA-aligned intervention planning and a **ChildAgent** to simulate heterogeneous child responses. The generated dialogues can be used for synthesis, analysis, and downstream distillation into smaller local models.

> **Clinical data notice**  
> Real clinical dialogue transcripts, identifiable child profiles, and derived private statistics are not included in this public release. The files under `data/*.sample.*` are synthetic placeholders for smoke testing and format reference only.

## 📊 Data Privacy
Due to potential data privacy and ethical risks, we are currently unable to publicly release real clinical dialogue data; however, we have open-sourced synthetic data. We appreciate your understanding.

## ✨ Highlights

- **O-T-A-C reasoning loop**: `DoctorAgent` follows Observe, Think, Act, and Correct stages for strategy-aware intervention.
- **ABA strategy control**: doctor turns are constrained to atomic labels such as `指令`, `强化`, `半辅助`, `全辅助`, and `其他`.
- **Probabilistic ChildAgent**: child response types are sampled from personal/global behavior statistics when available.
- **Correct module**: generated doctor utterances can be segmented and filtered so the final response better matches the intended ABA strategy.

## 📂 Repository Layout

```text
open_source/
├── src/asdagent/
│   ├── tinyaba/                 # Dual-agent dialogue synthesis
│   │   ├── main.py              # DoctorAgent, ChildAgent, World, CLI
│   │   ├── prompts/             # Prompt templates for O-T-A-C stages
│   │   ├── json_schema/         # Pydantic schemas for structured outputs
│   │   ├── convert_json2txt.py  # Convert generated JSONL sessions to TXT
│   ├── child_profile/           # Profile/statistics preprocessing utilities
│   └── utils/                   # OpenAI-compatible LLM API wrappers
├── data/                        # Synthetic sample inputs only
├── outputs/                     # Generated files, ignored by git
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

## ⚙️ Installation

```bash
cd LLMs/ASDAgent
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

## 🚀 Quick Start

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

## 📊 Data Format

Processed dialogue files use one utterance per line:

```text
医生: 你好包包，今天我们来学习元宵节与月亮。准备好了吗？（指令）
儿童: 月亮圆圆的。（相关的回答）
医生: 对，包包，月亮在元宵节的时候就是圆圆的，真厉害！（强化）
医生: 元宵节的时候，我们通常会吃一种圆圆的食物，你知道是什么吗？（指令）
儿童: 元宵。（相关的回答）
医生: 真聪明！元宵节的时候我们确实会吃圆圆的元宵！（强化）
医生: 你知道元宵节的晚上，我们可以看到什么其他有趣的东西吗？（指令）
儿童: 月亮。（相关的回答）
医生: 对啦，元宵节的晚上真的可以看到圆圆的月亮，你真棒！（强化）
医生: 除了圆圆的月亮，元宵节的晚上我们还可以看到什么呢？比如漂亮的灯笼？（指令）
儿童: 我想吃糖。（不相关的回答）
医生: 哦，你想吃糖呀，我听见了。（其他）
医生: 元宵节的时候，有时候我们还会看到我们做的灯笼，非常漂亮。除了灯笼，还可能看到什么特别的活动呢？（半辅助）
儿童: [儿童无响应]（无响应）
医生: 元宵节的晚上，除了灯笼，还有一种会发光的东西，（半辅助）
儿童: 烟花。（相关的回答）
医生: 真棒！你回答对了，元宵节的晚上确实会看到漂亮的烟花！（强化）
医生: 元宵节那天还有一些有趣的活动，你想知道有哪些吗？（指令）
儿童: 是什么活动呢？（相关的回答）
医生: 元宵节有猜灯谜的活动哦！灯谜是写在灯笼上的小谜题，你可以尝试去答对它们。你对这个活动感兴趣吗？（指令）
儿童: 为什么天空是蓝色的呢？（不相关的回答）
医生: 嗯，这也是个有趣的问题。（其他）
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

## 🧱 Core Components

**DoctorAgent**  
The doctor agent observes the latest child response, selects an intervention strategy, generates the next utterance, and optionally corrects strategy drift through segment-level filtering.

**ChildAgent**  
The child agent samples response types from personal and global statistics, then generates role-consistent child utterances through response-type-specific prompts.

**World**  
The world object coordinates turn-taking, memory, interruptions, persistence, and output paths.

## 🧩 Case Study

<img width="4024" height="2237" alt="image" src="https://github.com/user-attachments/assets/ceaeadcb-6961-4989-b75a-a694f62fb69d" />




## 📝 Citation

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

## ⚖️ License

This project is released under the [MIT License](LICENSE). The public code release does not include the private clinical dataset.
