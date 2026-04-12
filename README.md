# 🧩 [ACL 2026 Main] From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset

[![Venue: ACL 2026](https://img.shields.io/badge/Venue-ACL%202026%20Main-blue.svg)](https://2026.aclweb.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework: TRL](https://img.shields.io/badge/Framework-TRL-yellow)](https://github.com/huggingface/trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the ACL 2026 Main Conference paper: **"[From Synthesis to Clinical Assistance: A Strategy-Aware Agent Framework for Autism Intervention based on Real Clinical Dataset]"**.

## 📖 Overview

Early Intensive Behavioral Intervention (EIBI) based on Applied Behavior Analysis (ABA) is crucial for children with Autism Spectrum Disorder (ASD), but high-quality clinical data is extremely scarce due to privacy restrictions. 

**ASDAgent** introduces a novel "Synthesis-for-Distillation" paradigm. It utilizes a Dual-Agent sandbox (`DoctorAgent` and `ChildAgent`) to generate high-fidelity, strategy-aligned clinical dialogues. These synthetic dialogues are then used to distill therapeutic reasoning into deployable Small Language Models (SLMs).

### ✨ Core Features

* **O-T-A-C Reasoning Loop:** The `DoctorAgent` uses an explicit **Observe-Think-Act-Correct** mechanism. The `Correct` module acts as an adaptive safety filter (intercepting ~20-27% of strategy violations) to prevent "instruction stacking" and ensure strict adherence to ABA "Atomic Actions".
* **Dual-Agent Sandbox:** Simulates real-world interactions. The `ChildAgent` employs Probabilistic Behavior Modeling to reflect the multidimensional phenotypic heterogeneity of children with ASD.
* **Synthesis-for-Distillation:** Breaks the performance ceiling of pure real-data training. Mixing ASDAgent synthetic data with real data significantly boosts semantic generalization and strategy alignment for local SLMs (e.g., Qwen, Hunyuan, Llama).

## 📂 Repository Structure

```text
ASDAgent/
├── data/                 # Sample datasets and prompts (ASD-iLLM-8K subset)
├── src/
│   ├── agents/           # DoctorAgent (O-T-A-C) and ChildAgent implementations
│   ├── env/              # Sandbox environment for dual-agent interaction
│   └── utils/            # Data processing and evaluation scripts
├── scripts/
│   ├── run_simulation.sh # Script to generate synthetic data
│   ├── run_sft.sh        # Script for supervised fine-tuning (via TRL)
│   └── run_eval.sh       # Evaluation scripts (Multi-F1, LCS-F1, BERTScore)
├── requirements.txt
└── README.md
