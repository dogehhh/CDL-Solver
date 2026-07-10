#!/bin/bash

set -x

MODEL_PATH=/workspace/user_code/LLaMA-Factory/saves_new/qwen3vl-8b/full/sft_formalgeo7kv2_CoT_consCDL_0930_split

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=bzh666/Formalgeo7k_cot_conscdl_0930_split@train \
    data.val_files=bzh666/Formalgeo7k_cot_conscdl_0930_split@validation \
    data.prompt_key=problem\
    data.answer_key=answer\
    data.format_prompt=./examples/format_prompt/formalgeo.jinja\
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    worker.val_reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    trainer.experiment_name=qwen3_vl_8b_formalgeo7kv2_CoT_consCDL_0930_split \
    trainer.total_epochs=30 \
    worker.rollout.n=8\
    trainer.n_gpus_per_node=8
