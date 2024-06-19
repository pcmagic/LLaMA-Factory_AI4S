import multiprocessing

# args_list = [(111, ) for dataset_attr in dataset_attr_array]
args_list = [111 for i0 in range(10)]
def process_dataset_attr(args):
    # dataset_attr, model_args, data_args, stage = args
    # tokenizer = tokenizer_module["tokenizer"]
    # do_tokenization(dataset_attr, model_args, data_args, stage, tokenizer)
    pass
with multiprocessing.Pool(processes=2) as pool:
    pool.map(process_dataset_attr, args_list)