# CDL-Solver
[CVPR'26 Findings] Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving

#### Data


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
python tools/pseudo_class.py --cfg 'config/voc_train_ori_cfg.yaml' --model 'RECLIPPP'
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
