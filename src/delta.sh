#!/bin/bash
###
 # @Author: lihaitao
 # @Date: 2023-04-28 14:32:33
 # @LastEditors: Do not edit
 # @LastEditTime: 2023-12-22 00:38:37
 # @FilePath: /lht/SAILER_english/sailer.sh
### 

# Model name & Output Path
MODEL_NAME=test   # Use filename as model's output dir name
OUTPUT_DIR=results/$MODEL_NAME

if [ ! -d $OUTPUT_DIR/model ]; then
  mkdir -p $OUTPUT_DIR/model
  echo "makedir $OUTPUT_DIR/model"
fi

if [ ! -d $OUTPUT_DIR/logs ]; then
  mkdir -p $OUTPUT_DIR/logs
  echo "makedir $OUTPUT_DIR/logs"
fi

if [ ! -d $OUTPUT_DIR/tfboard/$MODEL_NAME ]; then
  mkdir -p $OUTPUT_DIR/tfboard/$MODEL_NAME
  echo "makedir $OUTPUT_DIR/tfboard/$MODEL_NAME"
fi


BATCH_SIZE_PER_GPU=12
GRAD_ACCU=3
LOCAL_RANK=0,1,2,3 \
CUDA_VISIBLE_DEVICES=1,5,6,8 python -m torch.distributed.launch \
  --use-env \
  --nproc_per_node 3 \
  --master_port 29508 \
  run_pretraining.py \
  --model_name_or_path xxxxx \
  --output_dir $OUTPUT_DIR/model/Delta \
  --do_train \
  --logging_steps 50 \
  --save_steps 500 \
  --fp16 \
  --logging_dir $OUTPUT_DIR/tfboard/$MODEL_NAME \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCU \
  --learning_rate 5e-6 \
  --overwrite_output_dir \
  --dataloader_drop_last \
  --dataloader_num_workers 4 \
  --max_seq_length 512 \
  --num_train_epochs 10 \
  --train_path all_text_train.json \
  --weight_decay 0.01 \
  --encoder_mask_ratio 0.15 \
  --decoder_mask_ratio 0.50 \
  --use_decoder_head \
  --enable_head_mlm \
  --ddp_find_unused_parameters True \
  --n_head_layers 1 \
  --info_nce_coef 1 \
  --temperature 2 \
  --top_align 0.65 \
  --n_deep_decoder_layers 6 \
  --translation_coef 1 \
  --model_save_path ../model/test