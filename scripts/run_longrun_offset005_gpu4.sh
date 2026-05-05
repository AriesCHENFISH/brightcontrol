#!/bin/bash
source /data1/cx/bright_venv/bin/activate
cd /data1/cx/brightcontrol
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 /data1/cx/brightcontrol/external/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="/data1/cx/sd15/" \
  --dataset_name="/data1/cx/brightcontrol/data/train_compare_6k/imagefolder" \
  --dataloader_num_workers=4 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/data1/cx/brightcontrol/train_logs/day2_best_method_longrun_offset005" \
  --checkpointing_steps=500 \
  --validation_prompt "a very dark photo of a cat" \
  --seed=42 \
  --report_to="tensorboard" \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --noise_offset=0.05