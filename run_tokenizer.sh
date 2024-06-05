python src/train_tokeinzer.py \
--model_name_or_path /home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/pretrain_ckp/gemma-1.3B \
--dataset_dir /home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/data/S1_all/100 \
--preprocessing_num_workers 80 \
--tokenized_path /home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/data/S1_all_test \
--output /home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/data/S1_all_test \
--seed 1 \
--stage pt
