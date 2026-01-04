"""
Optimized Training Commands for Kaggle - UIT ViQUAD 2.0
Copy and paste these commands into your Kaggle notebook cells
"""

# ============================================================================
# CELL 1: Setup and Installations (if needed)
# ============================================================================
"""
# Uncomment if needed
# !pip install transformers datasets evaluate accelerate -q
"""

# ============================================================================
# CELL 2: Training mBERT (Optimized)
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path google-bert/bert-base-multilingual-cased \
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
  --output_dir /kaggle/working/mbert-base \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0 \
  --overwrite_output_dir
"""

# ============================================================================
# CELL 3: Training PhoBERT (Optimized)
# Note: PhoBERT uses slow tokenizer - this is normal and code supports it
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path vinai/phobert-base \
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
  --output_dir /kaggle/working/phobert-base \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0 \
  --overwrite_output_dir
"""

# ============================================================================
# CELL 4: Training XLM-RoBERTa (Optimized)
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --train_file "/kaggle/input/uit-viquad-2-0/train.json" \
  --validation_file "/kaggle/input/uit-viquad-2-0/dev.json" \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --save_strategy steps \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 12 \
  --gradient_accumulation_steps 5 \
  --learning_rate 2e-5 \
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
  --output_dir /kaggle/working/xlmr-base \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0 \
  --overwrite_output_dir
"""

# ============================================================================
# CELL 5: Prediction on Test Set (mBERT)
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path /kaggle/working/mbert-base \
  --test_file "/kaggle/input/uit-viquad-2-0/Private_Test_ref.json" \
  --do_predict \
  --per_device_eval_batch_size 16 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataloader_num_workers 4 \
  --output_dir /kaggle/working/mbert-base/pred \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0
"""

# ============================================================================
# CELL 6: Prediction on Test Set (PhoBERT)
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path /kaggle/working/phobert-base \
  --test_file "/kaggle/input/uit-viquad-2-0/Private_Test_ref.json" \
  --do_predict \
  --per_device_eval_batch_size 16 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataloader_num_workers 4 \
  --output_dir /kaggle/working/phobert-base/pred \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0
"""

# ============================================================================
# CELL 7: Prediction on Test Set (XLM-RoBERTa)
# ============================================================================
"""
!python run_qa.py \
  --model_name_or_path /kaggle/working/xlmr-base \
  --test_file "/kaggle/input/uit-viquad-2-0/Private_Test_ref.json" \
  --do_predict \
  --per_device_eval_batch_size 12 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataloader_num_workers 4 \
  --output_dir /kaggle/working/xlmr-base/pred \
  --version_2_with_negative \
  --n_best_size 20 \
  --max_answer_length 30 \
  --null_score_diff_threshold 0.0
"""

# ============================================================================
# CELL 8: View Training Logs
# ============================================================================
"""
import os
import glob

# Find the latest log file
log_dir = "/kaggle/working/mbert-base/logs"
log_files = glob.glob(os.path.join(log_dir, "*.log"))
if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Latest log file: {latest_log}")
    with open(latest_log, 'r', encoding='utf-8') as f:
        print(f.read()[-5000:])  # Print last 5000 characters
"""

# ============================================================================
# CELL 9: Evaluate Results (Optional)
# ============================================================================
"""
# Run evaluation script if you have ground truth
!python evaluation.py \
  /kaggle/input/uit-viquad-2-0/ground_truth_private_test.json \
  /kaggle/working/mbert-base/pred/predict_predictions.json \
  --out-file /kaggle/working/mbert-base/eval_results.json
"""