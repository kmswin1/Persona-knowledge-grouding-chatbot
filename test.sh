#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
###decoder MTL
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_transformer-decoder_E2_L10 > test_log_focus/test_DECODER_MTL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_transformer-decoder_E2_L10 > test_log_focus/test_DECODER_MTL_ppl.log &

###encdec MTL
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_transformer-encdec_E2_L10 > test_log_focus/test_ENCDEC_MTL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_transformer-encdec_E2_L10 > test_log_focus/test_ENCDEC_MTL_ppl.log &

###gpt2 MTL
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10 > test_log_focus/test_GPT_MTL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10 > test_log_focus/test_GPT_MTL_ppl.log &

###bart MTL
# DEBUG
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --graph --constrain_fixed_mult 1.2 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v5.json > test_log_focus/test_BART_MTL_GRAPH_WO_REL_GENEVAL_1_2x.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --ner --constrain_fixed_mult 10.0 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v5.json > test_log_focus/test_BART_MTL_NER_V5_GENEVAL_10x.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --constrain_fixed_mult 5.0 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v5.json > test_log_focus/test_BART_MTL_LITERAL_GENEVAL_5x.log &

#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --literal 1.2 --graph 5.0 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v5.json > test_log_focus/test_BART_MTL_LITERAL_GRAPH_GENEVAL_1_2x_5x.log &

#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --without_rel --markov_graph 5.0 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v5.json > test_log_focus/test_BART_MTL_MARKOV_GRAPH_WO_REL_GENEVAL_5x.log &
CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 --beam_length_alpha 1.0 --markov_graph_v2 1.0 --max_length 100 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_LENGTH_100_ALPHA_1_0.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --beam_size 10 --markov_graph_v2 1.0 --max_length 40 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 --test_dataset_path data/new_valid_json_v6.json > test_log_focus/test_BART_MTL_BEAM_10_V6_LENGTH_40.log &

#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL_GENEVAL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL_ppl.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --gen_eval --constrain --graph --constrain_fixed_mult 1.5 --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL_GENEVAL_constrain_1_5x.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint ./models/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL_ppl.log &


###########ablations##########
###gpt2 LM
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM > test_log_focus/test_GPT_LM.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM > test_log_focus/test_GPT_LM_ppl.log &

###bart LM
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM > test_log_focus/test_BART_LM.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM > test_log_focus/test_BART_LM_ppl.log &


##gpt2 WOKS
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS > test_log_focus/test_GPT_wo_ks.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS > test_log_focus/test_GPT_wo_ks_ppl.log &

#
##bart WOKS
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS > test_log_focus/test_BART_wo_ks.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS > test_log_focus/test_BART_wo_ks_ppl.log &


##gpt2 WOPS
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS > test_log_focus/test_GPT_wo_ps.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS > test_log_focus/test_GPT_wo_ps_ppl.log &


##bart WOPS
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS > test_log_focus/test_BART_wo_ps.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS > test_log_focus/test_BART_wo_ps_ppl.log &


##########large models##########
##gpt2 medium
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large > test_log_focus/test_GPT2_large_MTL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large > test_log_focus/test_GPT2_large_MTL_ppl.log &

##bart large
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large > test_log/test_BART_large_MTL.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large > test_log/test_BART_large_MTL_ppl.log &

##gpt2 medium LM
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large_LM > test_log/test_GPT2_large_LM.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large_LM > test_log/test_GPT2_large_LM_ppl.log &

##bart large LM
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large_LM > test_log/test_BART_large_LM.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large_LM > test_log/test_BART_large_LM_ppl.log &

echo
