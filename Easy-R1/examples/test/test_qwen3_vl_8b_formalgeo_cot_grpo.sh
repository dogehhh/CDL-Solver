#!/bin/bash

set -x

MODEL_PATH=/workspace/user_code/EasyR1/checkpoints/easy_r1/qwen3_vl_8b_formalgeo7kv2_CoT_consCDL_0930_f1score/global_step_150/actor/huggingface

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=bzh666/Formalgeo7k_cot_conscdl_0814@train \
    data.val_files=bzh666/unigeo@train_cal_angle_gen_judge \
    data.prompt_key=problem\
    data.answer_key=answer\
    data.format_prompt=./examples/format_prompt/formalgeo.jinja\
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    worker.val_reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    trainer.experiment_name=qwen3_vl_8b_formalgeo7kv2_CoT_consCDL_0930_f1score\
    trainer.n_gpus_per_node=2\
    trainer.logger=["console"]\
    trainer.val_only=true
