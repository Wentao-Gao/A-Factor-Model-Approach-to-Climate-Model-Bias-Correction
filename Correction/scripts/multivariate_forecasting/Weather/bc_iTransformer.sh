export CUDA_VISIBLE_DEVICES=5,6,7

model_name=iTransformer


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path /RAID5/DataStorage/wentao/iTransformer/dataset/weather/withZ/merged_data_with_delta_prate.csv\
  --model_id weather_r1delta_36_1 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 36 \
  --pred_len 1 \
  --e_layers 3 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --target prate





