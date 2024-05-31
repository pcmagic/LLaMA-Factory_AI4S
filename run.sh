# # (1,0,0) (1,0,0)
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/model_pt.yaml;
# # (1,1,0) (0,1,0)
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/lora_0_pt.yaml;
# # (1,1,0) (1,1,0)
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/model_lora_0_pt.yaml;
# # (1,0,1) (0,0,1)
# cp saves/olmo/model_pt/config.json saves/olmo/model_lora_0_pt/model/;
# cp saves/olmo/model_pt/generation_config.json saves/olmo/model_lora_0_pt/model/;
# cp saves/olmo/lora_0_pt/adapter_config.json saves/olmo/model_lora_0_pt/lora_default/;
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/lora_1_pt.yaml;
# # (1,0,1) (1,0,1)
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/model_lora_1_pt.yaml;
# # (1,1,1) (1,1,1)
# cp saves/olmo/model_lora_0_pt/model/config.json saves/olmo/model_lora_1_pt/model/;
# cp saves/olmo/model_lora_0_pt/model/generation_config.json saves/olmo/model_lora_1_pt/model/;
# cp saves/olmo/lora_1_pt/adapter_config.json saves/olmo/model_lora_1_pt/lora_default/;
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_moe/model_lora_0_1_pt.yaml;
# cp saves/olmo/model_lora_1_pt/model/config.json saves/olmo/model_lora_0_1_pt/model/;
# cp saves/olmo/model_lora_0_pt/lora_default/adapter_config.json saves/olmo/model_lora_0_1_pt/lora_default/;
# cp saves/olmo/model_lora_1_pt/lora_default/adapter_config.json saves/olmo/model_lora_0_1_pt/lora_1/;



accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/model_pt.yaml;

accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/lora_0_pt.yaml;

accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/model_lora_0_pt.yaml;

cp saves/olmo/model_pt/config.json saves/olmo/model_lora_0_pt/model/;
cp saves/olmo/model_pt/generation_config.json saves/olmo/model_lora_0_pt/model/;
cp saves/olmo/lora_0_pt/adapter_config.json saves/olmo/model_lora_0_pt/lora_default/;

accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/lora_1_pt.yaml;

accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/model_lora_1_pt.yaml;

cp saves/olmo/model_lora_0_pt/model/config.json saves/olmo/model_lora_1_pt/model/;
cp saves/olmo/model_lora_0_pt/model/generation_config.json saves/olmo/model_lora_1_pt/model/;
cp saves/olmo/lora_1_pt/adapter_config.json saves/olmo/model_lora_1_pt/lora_default/;

accelerate launch --config_file examples/accelerate/single_config.yaml src/train.py examples/lora_moe/model_lora_0_1_pt.yaml;

cp saves/olmo/model_lora_1_pt/model/config.json saves/olmo/model_lora_0_1_pt/model/;
cp saves/olmo/model_lora_0_pt/lora_default/adapter_config.json saves/olmo/model_lora_0_1_pt/lora_default/;
cp saves/olmo/model_lora_1_pt/lora_default/adapter_config.json saves/olmo/model_lora_0_1_pt/lora_1/;
