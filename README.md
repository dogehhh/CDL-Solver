# CDL-Solver
[CVPR'26 Findings] Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving

#### Data
[Formalgeo-Rec-CoT]([[https://drive.google.com/file/d/1FjTNj8PPGlce4xAQIXwjhi020glX3fSW/view?usp=drive_link](https://huggingface.co/datasets/bzh666/Formalgeo7k_cot_conscdl_0930)]) 

#### Training

##### Stage 1 SFT Stage

```python
# We use LLaMA-Factory for SFT stages
cd LLaMA-Factory
llamafactory-cli train examples/train_full/qwen3vl_8b_full_sft.yaml
```

##### Stage 2 RL Stage

```python
# We use Easy-R1 for RL stages
cd Easy-R1
bash examples/qwen3_vl_8b_formalgeo_cot_grpo.sh
```

#### Citing

Please cite our paper if you use our code in your research:

```
@InProceedings{Wang_2026_CVPR,
    author    = {Wang, Jingyun and Li, Dian and Wang, Xiaohan and Liu, Gang and Yan, Jiahong and Kang, Guoliang},
    title     = {Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometric Problem Solving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {5958-5967}
}
```
