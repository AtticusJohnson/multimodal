

export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
--workerGPU=1 \
--mode=export \
--train_input_fp=train_list.list_csv \
--eval_input_fp=train_list.list_csv \
--predict_input_fp=train_list.list_csv \
--pretrain_model_name_or_path=amazon_best_model/model.ckpt-2500 \
--input_sequence_length=64 \
--train_batch_size=1 \
--num_epochs=1 \
--model_dir=amazon_model_dir_20220313 \
--learning_rate=1e-4 \
--image_feature_size=131072 \
--input_schema=image_feature:float:131072,image_mask:int:64,masked_patch_positions:int:5,input_ids:int:64,input_mask:int:64,segment_ids:int:64,masked_lm_positions:int:10,masked_lm_ids:int:10,masked_lm_weights:float:10,nx_sent_labels:int:1 \
--output_dir=./amazon_ouput \
--predict_checkpoint_path=amazon_best_model/model.ckpt-2500 \
--predict_batch_size=16
