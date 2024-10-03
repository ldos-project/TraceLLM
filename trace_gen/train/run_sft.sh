#!/bin/bash
accelerate launch scripts/sft.py \
  --model_ckpt meta-llama/Llama-2-7b-hf-sft \
  --train_batch_size 1 \
  --valid_batch_size 1 \
  --learning_rate 1e-4 \
  --gradient_accumulation 1 \
  --gradient_checkpointing False \
  --save_checkpoint_steps 200 \
  --save_dir <FILLME>
