from types import MethodType
from typing import TYPE_CHECKING, Dict, Optional

from transformers import Trainer

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from collections import defaultdict
import os
import torch
from pathlib import Path


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

        if self.finetuning_args.finetuning_type == "model_lora":
            model_output_dir = os.path.join(output_dir,'model')
            os.makedirs(model_output_dir, exist_ok=True)
            self.model.base_model.model.save_pretrained(model_output_dir)
            self.tokenizer.save_pretrained(model_output_dir)

        if self.finetuning_args.link_latest:
            latest_path = Path('/'.join(output_dir.split('/')[:-1])) / "latest"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(output_dir.split('/')[-1], target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != output_dir:
                    raise

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if self.processor is not None:
    #         output_dir = output_dir if output_dir is not None else self.args.output_dir
    #         getattr(self.processor, "image_processor").save_pretrained(output_dir)
    #
    #     # 改写trainer的save_model，在model_lora的时候分别保存model和lora权重
    #     from transformers.trainer import TRAINING_ARGS_NAME
    #     import safetensors.torch
    #
    #     output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #     #
    #     # if self.tokenizer is not None:
    #     #     self.tokenizer.save_pretrained(output_dir)
    #
    #     lora_dict= defaultdict(dict)
    #     model_dict = {}
    #     for k, v in self.model.named_parameters():
    #         if 'lora' in k:
    #             lora_dict[k.split('.')[-2]][k] = v
    #         else:
    #             model_dict[k] = v
    #     if self.finetuning_args.finetuning_type == "model_lora":
    #         os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    #
    #         if self.tokenizer is not None:
    #             self.tokenizer.save_pretrained(os.path.join(output_dir, 'model'))
    #         torch.save(self.args, os.path.join(output_dir, 'model', TRAINING_ARGS_NAME))
    #
    #         for k, v in lora_dict.items():
    #             os.makedirs(os.path.join(output_dir, f'lora_{k}'), exist_ok=True)
    #             if self.tokenizer is not None:
    #                 self.tokenizer.save_pretrained(os.path.join(output_dir, f'lora_{k}'))
    #             torch.save(self.args, os.path.join(output_dir, f'lora_{k}', TRAINING_ARGS_NAME))
    #
    #             safetensors.torch.save_file(
    #                     v, os.path.join(output_dir, f'lora_{k}', f"adapter_model.safetensors"), metadata={"format": "pt"}
    #                 )
    #         safetensors.torch.save_file(
    #                 model_dict, os.path.join(output_dir, 'model', "model.safetensors"), metadata={"format": "pt"}
    #             )
    #         # self.model.save_pretrained(
    #         #      os.path.join(output_dir, 'model'), state_dict=model_dict, safe_serialization=self.args.save_safetensors
    #         # )
    #     elif self.finetuning_args.finetuning_type == "lora":
    #         # for k, v in lora_dict.items():
    #         # os.makedirs(os.path.join(output_dir, f'lora_{k}'), exist_ok=True)
    #         if self.tokenizer is not None:
    #             self.tokenizer.save_pretrained(output_dir)
    #         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #
    #         # safetensors.torch.save_file(
    #         #         v, os.path.join(output_dir, f'lora_{k}', "adapter_model.safetensors"), metadata={"format": "pt"}
    #         #     )
    #         self.model.save_pretrained(
    #              output_dir, state_dict=lora_dict['default'], safe_serialization=self.args.save_safetensors
    #         )
    #     else:
    #         if self.tokenizer is not None:
    #             self.tokenizer.save_pretrained(os.path.join(output_dir))
    #         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #         # safetensors.torch.save_file(
    #         #         model_dict, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"}
    #         #     )
    #         self.model.save_pretrained(
    #              output_dir, state_dict=model_dict, safe_serialization=self.args.save_safetensors
    #         )
