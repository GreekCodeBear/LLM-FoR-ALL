import random
import warnings
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, PreTrainedModel

from trl import SFTTrainer, SFTConfig
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache,DataCollatorForChatML
from trl.models import PreTrainedModelWrapper

from dataclasses import dataclass
from accelerate.utils import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed

@dataclass
class DistillConfig(SFTConfig):
    """
    Configuration class for DistillConfig.
    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        alpha (`float`, *optional*, defaults to `0.5`):
            Alpha parameter that controls the importance of the KL divergence term in the loss.
        max_new_tokens (`int`, *optional*, defaults to `1024`):
            Maximum number of tokens to generate per completion.
    """

    temperature: float = 0.9
    alpha: float = 1
    max_new_tokens: int = 1024

    def __post_init__(self):
        super().__post_init__()
        # check alpha is in the range [0, 1]
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in the range [0.0, 1.0].")

class DistillTrainer(SFTTrainer):
    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[DistillConfig] = None,
        *sft_args,
        **kwargs,
    ):
        args.remove_unused_columns = False
        kwargs["data_collator"] = DataCollatorForChatML(tokenizer=kwargs["tokenizer"], max_length=args.max_seq_length)
        super().__init__(*sft_args, args=args, **kwargs)
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.alpha = args.alpha
        self.temperature = args.temperature
        
        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)


        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        student_loss = outputs_student.loss

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        # shifted_labels = inputs["labels"][:, prompt_lengths:]

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(shifted_student_logits / self.temperature, dim=-1),
                F.softmax(shifted_teacher_logits / self.temperature, dim=-1),
            )
            * (self.temperature ** 2)
        )
        # Return weighted student loss
        loss =  self.alpha *student_loss + (1.0 - self.alpha) * loss_logits

        # empty cache
        empty_cache()
        
        # Return loss
        return (loss, outputs_student) if return_outputs else loss
    
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model