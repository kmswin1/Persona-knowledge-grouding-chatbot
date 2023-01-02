#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

#Train the model only with LM
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name GPT2 --incontext > train_log_focus/e2_L20_LM_gpt.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name BART --incontext > train_log_focus/e2_L20_LM_bart.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name transformer-decoder --incontext > train_log_focus/e2_L20_LM_decoder.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_LM --ps_coef 0 --kn_coef 0 --model_name transformer-encdec --incontext > train_log_focus/e2_L20_LM_encdec.log &

#Train the model with Knowledge Selection, Persona Selection and Language Modeling
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name GPT2 --incontext > train_log_focus/e2_L20_gpt.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name BART --incontext > train_log_focus/e2_L20_bart.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name transformer-decoder --incontext > train_log_focus/e2_L20_decoder.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10 --model_name transformer-encdec --incontext > train_log_focus/e2_L20_encdec.log &

CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_V4_KS_TRIPLE_APPEND --append_k_triples --model_name BART \
    --train_dataset_path data/lines_v4_train.json \
    --train_dataset_cache data/focus_train_cache_V4.tar.gz \
    --dev_dataset_path data/lines_v4_valid.json --dev_dataset_cache data/focus_valid_cache_V4.tar.gz \
    --incontext > train_log_focus/e2_L20_V4_KS_TRIPLE_APPEND_bart.log &

# Train model for development
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_V5_KS_TRIPLE --model_name BART --use_k_triples --train_dataset_path dev_data/lines_v5_train.json \
#    --train_dataset_cache dev_data/focus_train_cache_V5.tar.gz --dev_dataset_path dev_data/lines_v5_valid.json --dev_dataset_cache dev_data/focus_cache_V5.tar.gz \
#    --incontext > train_log_focus/e2_L20_bart.log &

#Train the model w/o Knowledge Selection
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --kn_coef 0 --flag E2_L10_WO_KS --model_name BART --incontext > train_log_focus/e2_L20_bart_WO_KS.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --kn_coef 0 --flag E2_L10_WO_KS --model_name GPT2 --incontext > train_log_focus/e2_L20_gpt_WO_KS.log &

#Train the model w/o Persona Selection
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --ps_coef 0 --flag E2_L10_WO_PS --model_name GPT2 --incontext > train_log_focus/e2_L20_gpt_WO_PS.log &
#CUDA_VISIBLE_DEVICES=0 nohup python3 train_focus.py --n_epochs 2 --lm_coef 10 --ps_coef 0 --flag E2_L10_WO_PS --model_name BART --incontext > train_log_focus/e2_L20_bart_WO_PS.log &
