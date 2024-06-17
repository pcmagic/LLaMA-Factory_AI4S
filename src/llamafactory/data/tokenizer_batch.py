import glob
import os
import sys

# sys.path.append(r"/home/ubuntu/Desktop/github/LLaMA-Factory_AI4S/src/llamafactory")
from ..model.loader import load_tokenizer
# from ..hparams.model_args import ModelArguments
# from ..hparams.data_args import DataArguments
from ..hparams import get_train_args
from .template import get_template_and_fix_tokenizer
from .loader import *
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..extras.constants import FILEEXT2TYPE

# from transformers import Seq2SeqTrainingArguments
import natsort 
from datasets import Dataset
import numpy as np


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

logger = get_logger(__name__)


def tokenizer_batch(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        args)

    processor = data_args.preprocessing_num_workers

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    ddir = data_args.dataset_dir
    datalist = []
    new_ddir = ""
    for dl in ddir:
        if dl.startswith("PRETRAIN_DATA_PATH"):
            dl = os.environ.get("PRETRAIN_DATA_PATH") + \
                dl.replace("PRETRAIN_DATA_PATH", "")
        new_ddir += dl + ","
        for file_name in glob.glob(os.path.join(dl, "*")):
            data_type = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            if data_type is not None and "dataset_info" not in file_name:
                datalist.append(file_name)
    data_args.dataset = datalist
    data_args.dataset_dir = new_ddir

    idx = 0
    all_datasets = []
    all_datasets_name = []
    stage = "pt"
    dataset_attr_array = np.array(get_dataset_list(data_args))
    dataset_sort_idx = natsort.index_natsorted([str(i0) for i0 in dataset_attr_array])
    dataset_attr_array = dataset_attr_array[dataset_sort_idx]
    logger.info(dataset_attr_array)
    for idx, dataset_attr in enumerate(dataset_attr_array):
        lsd = load_single_dataset(dataset_attr, model_args, data_args, stage)
        if lsd is None:
            continue
        all_datasets_name.append(dataset_attr)
        logger.info(dataset_attr)
        logger.info(lsd)
        all_datasets.append(lsd)
        if (idx + 1) % 10 == 0 or idx == len(dataset_attr_array) - 1:
            dataset = merge_dataset(all_datasets, data_args, training_args)
            preprocess_func, _ = get_preprocess_and_print_func(
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
            # logger.info(f'Total number of data files: {len(dataset)}')
            dataset = dataset.map(
                preprocess_func, batched=True, remove_columns=column_names, **kwargs)
            # logger.info(f'Total number of data files after tokenizer: {len(dataset)}\n\n\n')
            tokenized_path = data_args.tokenized_path + \
                "_" + str((idx + 10) // 10)
            dataset = Dataset.from_list(list(dataset))
            dataset.save_to_disk(tokenized_path)
            logger.info("Include datasets: \n" + str(all_datasets_name))
            logger.info("Tokenized dataset saved at {}.".format(
                data_args.tokenized_path))
            logger.info(
                "Please restart the training with `--tokenized_path {}`.".format(data_args.tokenized_path))
            all_datasets = []
