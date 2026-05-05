#!/bin/bash
source /data1/cx/bright_venv/bin/activate
cd /data1/cx/brightcontrol

# Configuration
export MODEL_NAME="/data1/cx/sd15"
export TRAIN_DIR="/data1/cx/brightcontrol/data/train_compare_6k/imagefolder"
export OUT_DIR="/data1/cx/brightcontrol/train_logs/day2_best_method_longrun_offset005"

# Create output directory
mkdir -p "$OUT_DIR"

echo "Starting long-run training with noise_offset=0.05 on GPU4"
echo "Model: $MODEL_NAME"
echo "Train data: $TRAIN_DIR"
echo "Output: $OUT_DIR"
echo "Max steps: 2500"
echo "Checkpoint steps: 500"

# More representative validation prompts (covering different lighting conditions)
VALIDATION_PROMPTS=(
    "a candle-lit dinner table in a very dark room"
    "a snowy field under harsh midday sunlight, extremely bright"
    "a white ceramic mug product photo on a pure white background"
    "a luxury watch product photo on a pure black background"
    "a person standing in front of a bright window, strong backlighting, silhouette"
    "a rainy city street at night with neon reflections, very dark scene"
    "a cozy living room in balanced natural lighting"
    "a red apple on a wooden table in normal indoor lighting"
)

# Create validation prompts file
VALIDATION_FILE="$OUT_DIR/validation_prompts.txt"
printf "%s\n" "${VALIDATION_PROMPTS[@]}" > "$VALIDATION_FILE"
echo "Validation prompts saved to: $VALIDATION_FILE"

# Start training
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 --main_process_port=29501 \
  /data1/cx/brightcontrol/external/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="$TRAIN_DIR" \
  --dataloader_num_workers=4 \
  --image_column=image \
  --caption_column=text \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2500 \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --validation_prompt="a white ceramic mug product photo on a pure white background" \
  --num_validation_images=4 \
  --validation_epochs=1 \
  --rank=8 \
  --seed=42 \
  --noise_offset=0.05 \
  --mixed_precision=fp16 \
  --enable_xformers_memory_efficient_attention \
  --output_dir="$OUT_DIR" \
  --report_to=tensorboard 2>&1 | tee "$OUT_DIR/train.log"

echo "Training completed. Log saved to: $OUT_DIR/train.log"