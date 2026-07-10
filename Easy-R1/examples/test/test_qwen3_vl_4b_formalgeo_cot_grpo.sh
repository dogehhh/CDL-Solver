#!/bin/bash

set -x

export VLLM_ATTENTION_BACKEND=FLASHINFER

MODEL_PATH=/workspace/user_code/LLaMA-Factory/saves/qwen3vl-4b/full/sft_formalgeo7kv2_CoT_consCDL_0930

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=bzh666/Formalgeo7k_cot_conscdl_0814@train \
    data.val_files=bzh666/mathvista_geo@val \
    data.prompt_key=problem\
    data.answer_key=answer\
    data.format_prompt=./examples/format_prompt/formalgeo.jinja\
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    worker.val_reward.reward_function=./examples/reward_function/cdl_cot.py:compute_score \
    trainer.experiment_name=qwen3_vl_4b_formalgeo7kv2_CoT_consCDL_0930 \
    trainer.n_gpus_per_node=2\
    trainer.logger=["console"]\
    trainer.val_only=true
