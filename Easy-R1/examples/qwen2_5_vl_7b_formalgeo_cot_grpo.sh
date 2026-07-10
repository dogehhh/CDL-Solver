#!/bin/bash

set -x

MODEL_PATH=/workspace/user_code/LLaMA-Factory/saves_new/qwen2_5vl-7b/full/sft_formalgeo7kv2_CoT_consCDL_0930

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=bzh666/Formalgeo7k_cot_conscdl_0930@train \
    data.val_files=bzh666/Formalgeo7k_cot_conscdl_1010@validation \
    data.prompt_key=problem\
    data.answer_key=answer\
    data.format_prompt=./examples/format_prompt/formalgeo.jinja\
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    worker.val_reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_formalgeo7kv2_CoT_consCDL_and_imgCDL \
    worker.rollout.n=8\
    trainer.n_gpus_per_node=8
