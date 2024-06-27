#!/usr/bin/env bash
 
export CUDA_VISIBLE_DEVICES=0

root_path="~/work/strategies/ai/data/csv_data/"
data_path="y888_31m_2020-01-06 14:42:00_to_2022-12-13 10:39:00.csv"
# data_path="y888_31m_2010-01-07 14:57:00_to_2022-12-13 10:39:00.csv" 

model_name=TimeMixer

seq_len=768
pred_len=48
label_len=48
e_layers=4
down_sampling_layers=4
down_sampling_window=2
learning_rate=0.01
d_model=26
d_ff=32
batch_size=256
train_epochs=20
patience=10

python run_walter_.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  $root_path \
  --data_path  "$data_path" \
  --model_id walter \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
