<div align="center">

# 🧩 CDL-Solver

### Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving

[![Conference](https://img.shields.io/badge/CVPR'26-Findings-1f6feb?style=flat-square)](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_Concise_Geometric_Description_as_a_Bridge_Unleashing_the_Potential_of_CVPRF_2026_paper.html)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-ffca28?style=flat-square)](https://huggingface.co/datasets/bzh666/Formalgeo7k_cot_conscdl_0930)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## 📦 Data

| Dataset | Description | Link |
| :--- | :--- | :---: |
| **Formalgeo-Rec-CoT** | Formalgeo7k with concise CDL Chain-of-Thought annotations | [🤗 HuggingFace](https://huggingface.co/datasets/bzh666/Formalgeo7k_cot_conscdl_0930) |

## 🏋️ Model Weights

We release the following model checkpoints:

| Base Model | Stage | Link |
| :--- | :---: | :---: |
| **Qwen3-VL-8B** | SFT | [🤗 HuggingFace](https://huggingface.co/bzh666/sft_formalgeo7kv2_CoT_consCDL_0930_qwen3-vl8b) |
| **Qwen3-VL-8B** | RL | [🤗 HuggingFace](https://huggingface.co/bzh666/qwen3_vl_8b_formalgeo7kv2_CoT_consCDL_0930_f1score) |
<!-- | **Qwen2.5-VL-7B** | SFT | [🤗 HuggingFace]() |
| **Qwen2.5-VL-7B** | RL | [🤗 HuggingFace]() | -->

## 🚀 Training

### 🔹 Stage 1 — SFT (Supervised Fine-Tuning)

```bash
# We use LLaMA-Factory for SFT stages
cd LLaMA-Factory
llamafactory-cli train examples/train_full/qwen3vl_8b_full_sft.yaml
```

### 🔹 Stage 2 — RL (Reinforcement Learning)

```bash
# We use Easy-R1 for RL stages
cd Easy-R1
bash examples/qwen3_vl_8b_formalgeo_cot_grpo.sh
```

## 📖 Citing

If you find our work useful, please consider citing our paper:

```bibtex
@InProceedings{Wang_2026_CVPR,
    author    = {Wang, Jingyun and Li, Dian and Wang, Xiaohan and Liu, Gang and Yan, Jiahong and Kang, Guoliang},
    title     = {Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometric Problem Solving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {5958-5967}
}
```
