# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .trainer import CustomTrainer
from pathlib import Path
from filelock import FileLock
import os
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # #####################################
    # # check params is trainable or not ##
    # from prettytable import PrettyTable
    # def count_parameters(net):
    #     table = PrettyTable(["Modules", "Parameters", "Trainable"])
    #     total_params = 0
    #     for name, parameter in net.named_parameters():
    #         if not parameter.requires_grad:
    #             tr = 'false'
    #         else:
    #             tr = 'true'
    #         params = parameter.numel()
    #         table.add_row([name, params, tr])
    #         total_params += params
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params
    #
    # count_parameters(model)
    # raise
    # #####################################

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        output_dir = os.path.join(training_args.output_dir, 'final')
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

        if finetuning_args.link_latest:
            latest_path = Path(training_args.output_dir) / "latest"
            lock_path = latest_path.with_suffix('.lock')

            with FileLock(lock_path): 
                latest_path.unlink(missing_ok=True)
                try:
                    latest_path.symlink_to('final', target_is_directory=True)
                except FileExistsError:
                    # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                    # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                    # file-systems.
                    if latest_path.resolve().name != 'final':
                        raise

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
