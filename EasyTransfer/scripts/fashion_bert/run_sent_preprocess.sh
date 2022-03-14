BERT_BASE_DIR=/dataset/atticus/google-bert/uncased_L-12_H-768_A-12

python bert/create_pretraining_data.py \
  --input_file=bert/sample_text.txt \
  --output_file=./tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=64 \
  --max_predictions_per_seq=10 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

