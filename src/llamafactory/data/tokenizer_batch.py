import glob
import os
import sys
#sys.path.append(r"/home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/src/llamafactory")
from ..model.loader import load_tokenizer
from ..hparams.model_args import ModelArguments
from ..hparams.data_args import DataArguments
from ..hparams import get_train_args
from .template import get_template_and_fix_tokenizer
from .loader import *
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from transformers import Seq2SeqTrainingArguments



def tokenizer_batch(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    # 下面流程中涉及到的要配置参数
    # model_args = ModelArguments()
    # #tokenizer的配置项
    # model_args.model_name_or_path = ''
    # # model_args.hf_hub_token = ''
    # # model_args.cache_dir = ''
    # # model_args.use_fast_tokenizer = ''
    # # model_args.split_special_tokens = ''
    # # model_args.model_revision = ''
    #
    # #Data的配置项
    # data_args = DataArguments()
    # data_args.dataset_dir = ''
    # data_args.preprocessing_num_workers = 80
    # data_args.tokenized_path = ''
    # # data_args.template = ''
    #
    # #Train的配置项
    # training_args = Seq2SeqTrainingArguments()
    # training_args.seed = 1

    processor = 64

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    ddir = data_args.dataset_dir.split(',')
    datalist = []
    new_ddir = ''
    for dl in ddir:
        if dl.startswith('PRETRAIN_DATA_PATH'):
            dl = os.environ.get('PRETRAIN_DATA_PATH') + dl.replace('PRETRAIN_DATA_PATH', '')
        new_ddir += dl + ','
        datalist += [i for i in glob.glob(os.path.join(dl, '*'))]
        datalist.remove(os.path.join(dl, 'dataset_info.json'))
    data_args.dataset = ','.join(datalist)
    data_args.dataset_dir = new_ddir

    idx = 0
    all_datasets = []
    stage = "pt"
    dataset_attr_array = get_dataset_list(data_args)
    while idx < len(dataset_attr_array):
        dataset_attr = dataset_attr_array[idx]
        lsd = load_single_dataset(dataset_attr, model_args, data_args, stage)
        if lsd is None:
            continue
        all_datasets.append(lsd)
        if (idx + 1) % 10 == 0 or idx == len(dataset_attr_array) - 1:
            dataset = merge_dataset(all_datasets, data_args, training_args)
            preprocess_func, print_function = get_preprocess_and_print_func(
                data_args, training_args, stage, template, tokenizer, processor
            )
            column_names = list(next(iter(dataset)).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache),
                    desc="Running tokenizer on dataset",
                )
            print(f'Total number of data files: {len(dataset)}')
            dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
            print(f'Total number of data files after tokenizer: {len(dataset)}\n\n\n')
            tokenized_path = data_args.tokenized_path + "_" + str((idx + 10)//10)
            dataset.save_to_disk(tokenized_path)
            logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
            logger.info(
                "Please restart the training with `--tokenized_path {}`.".format(data_args.tokenized_path))
            all_datasets = []
        idx += 1

if __name__ == "__main__":
    main()
