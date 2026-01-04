#!/bin/bash
# Training script for PhoBERT on Kaggle - UIT ViQUAD 2.0
# Note: PhoBERT uses slow tokenizer, but code has been updated to support it

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

MODEL_NAME="vinai/phobert-base"
OUTPUT_DIR="/kaggle/working/phobert-base"

echo "Starting PhoBERT training (using slow tokenizer - this is normal)..."

python run_qa.py \
  --model_name_or_path ${MODEL_NAME} \
  --train_file "/kaggle/input/uit-viquad-2-0/train.json" \
  --validation_file "/kaggle/input/uit-viquad-2-0/dev.json" \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --save_strategy steps \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --fp16 \
  --dataloader_num_workers 4 \
  --logging_steps 100 \
  --save_steps 500 \
  --eval_steps 500 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --greater_is_better true \
  --save_total_limit 3 \
  --seed 42 \
  --weight_decay 0.01 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --output_dir ${OUTPUT_DIR} \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0 \
  --overwrite_output_dir

echo "PhoBERT training completed. Best model saved in ${OUTPUT_DIR}"

# Prediction on test set
echo "Starting prediction on test set..."

python run_qa.py \
  --model_name_or_path ${OUTPUT_DIR} \
  --test_file "/kaggle/input/uit-viquad-2-0/Private_Test_ref.json" \
  --do_predict \
  --per_device_eval_batch_size 16 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR}/pred \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0

echo "Prediction completed. Results saved in ${OUTPUT_DIR}/pred"

