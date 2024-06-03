from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
config = AutoConfig.from_pretrained('google/gemma-2b', vocab_size=len(tokenizer))
config.hidden_activation = config.hidden_act
# for gemma 1.3B
# config.hidden_size = 1024
model = AutoModelForCausalLM.from_config(config)

model.save_pretrained('pretrain_ckp/gemma1.3/')
tokenizer.save_pretrained('pretrain_ckp/gemma1.3/')
