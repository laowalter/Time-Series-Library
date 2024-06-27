#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

root_path="~/work/strategies/ai/data/csv_data/"
# data_path="y888_31m_2020-01-06 14:42:00_to_2022-12-13 10:39:00.csv"
# data_path="y888_31m_2010-01-07 14:57:00_to_2022-12-13 10:39:00.csv"
data_path="m888_31m_2010-01-07 14:57:00_to_2022-12-13 11:27:00.csv"
# data_path="rb888_31m_2010-01-07 14:57:00_to_2022-11-25 14:57:00.csv"

model_name=TimesNet
seq_len=768
pred_len=48
label_len=48
e_layers=2
d_layers=1

python -u run_walter_.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --batch_size 128 \
  --root_path $root_path \
  --data_path "$data_path" \
  --model_id Exchange_walter_data \
  --model $model_name \
  --freq 't' \
  --data custom \
  --features MS \
  --inverse \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len\
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --d_model 128 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1
