# PLEASE ADJUST SEED with --seed .

# GREEDY MAX LEN
CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 1 \
--markov_graph_v2 1.0 --max_length 20 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_GREEDY_V6_MAXLEN_20.log &


CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 1 \
--markov_graph_v2 1.0 --max_length 40 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_GREEDY_V6_MAXLEN_40.log &

# BEAM MAXLEN

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --max_length 20 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MAXLEN_40.log &


CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --max_length 40 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MAXLEN_40.log &

# BEAM MINLEN

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --min_length 5 --max_length 100 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MINMAXLEN_5_100.log &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --min_length 10 --max_length 100 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MINMAXLEN_10_100.log &

# BEAM ALPHA ADJ

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --beam_length_alpha 1.0 --max_length 100 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MINMAXLEN_10_100_ALPHA_1_0.log &

CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 \
--markov_graph_v2 1.0 --beam_length_alpha 0.8 --max_length 100 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 \
--test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_MINMAXLEN_10_100_ALPHA_0_8.log &