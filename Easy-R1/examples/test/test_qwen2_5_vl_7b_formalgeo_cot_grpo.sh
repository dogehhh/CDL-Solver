#!/bin/bash

set -x

MODEL_PATH=/workspace/user_code/EasyR1_old/checkpoints/easy_r1/qwen2_5_vl_7b_formalgeo7kv2_CoT_consCDL_wo_cons_format/global_step_150/actor/huggingface

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=bzh666/Formalgeo7k_cot_conscdl_0814@train \
    data.val_files=bzh666/unigeo@val_cal \
    data.prompt_key=problem\
    data.answer_key=answer\
    data.format_prompt=./examples/format_prompt/formalgeo.jinja\
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    worker.val_reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_formalgeo7kv2_CoT_consCDL_wo_cons_format \
    trainer.n_gpus_per_node=2\
    trainer.logger=["console"]\
    trainer.val_only=true
