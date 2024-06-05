from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# current version inspired from google/gemma-2b
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
config = AutoConfig.from_pretrained('google/gemma-2b', vocab_size=len(tokenizer))
config.hidden_activation = config.hidden_act
# dbg code, generate a small checkpoint for dbg. 
config.hidden_size = 128
config.num_hidden_layers = 2
model = AutoModelForCausalLM.from_config(config)

model.save_pretrained('pretrain_ckp/dbg_gemma1.3/')
tokenizer.save_pretrained('pretrain_ckp/dbg_gemma1.3/')
