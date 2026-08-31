# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
import sys
import torch.nn.functional as F
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # NOTE: The "ar" loss assumes an SDAR model that internally concatenates [noisy|clean],
        # producing hidden_states of length 2*seq_len. This will not work with standard AR models.
        if self.finetuning_args.idlm_loss_type != "ar":
            return super().compute_loss(model, inputs, *args, **kwargs)

        alpha = self.finetuning_args.ce_alpha
        labels = inputs.get("labels")
        assert labels is not None

        # The collator doesn't emit position_ids for non-packed data. Generate them
        # here so both the model's internal use and our relay canvas see the same thing.
        if inputs.get("position_ids") is None:
            _b, _l = inputs["input_ids"].shape
            inputs["position_ids"] = (
                torch.arange(_l, device=inputs["input_ids"].device)
                .unsqueeze(0).expand(_b, -1).contiguous()
            )

        # Forward pass — the SDAR model internally concatenates [noisy_xt | clean_x0],
        # so input_ids has length L but hidden_states has length 2L.
        

        unwrapped_model = getattr(model, "module", model)
        seq_len = inputs["input_ids"].size(-1)
        use_relay = getattr(self.finetuning_args, "use_relay", False)

        prev_latent = None
        bd = None
        n_pred = None

        if use_relay:
            # PEFT wraps the model, so type(unwrapped_model).__module__ is peft.peft_model.
            # Find the already-imported modeling_sdar module directly instead.
            sdar = next(m for n, m in sys.modules.items() if n.endswith(".modeling_sdar"))

            # ---- pass 1: all-mask, no grad. We only want the hidden states. ----
            with torch.no_grad():
                out1 = model(**inputs, output_hidden_states=True)
            h = out1["hidden_states"][-1][:, :seq_len, :]
            h = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)   # hidden[i] predicts i+1
            if getattr(self.finetuning_args, "stop_grad_relay", True):
                h = h.detach()
            prev_latent = torch.cat([h, torch.zeros_like(h)], dim=1)  # zeros on clean half

            # ---- build the pass-2 canvas ----
            pos = sdar.modify_padded_position_ids_2d(inputs["position_ids"].clone())
            bd = list(unwrapped_model.prepare_for_bd_training(
                inputs["input_ids"], pos, (labels == -100)))
            concat_ids, concat_pos, attn, keep_half, keep, p_mask = bd

            bsz = inputs["input_ids"].size(0)
            blk = unwrapped_model.config.block_size
            dev = concat_ids.device

            # random rejection boundary per block. "<=" reveals the correction too.
            nblk   = seq_len // blk
            r = torch.randint(0, blk - 1, (bsz, nblk), device=dev)
            off    = torch.arange(blk, device=dev).view(1, 1, -1)
            reveal = (off <= r.unsqueeze(-1)).reshape(bsz, nblk * blk)
            if reveal.size(1) < seq_len:
                reveal = F.pad(reveal, (0, seq_len - reveal.size(1)), value=False)
            reveal = reveal & keep_half        # only touch positions that are masked

            # noisy and clean halves are interleaved per packed sub-sequence
            num_tokens = sdar.calculate_token_nums(pos)
            if self.state.global_step == 0:
                logger.info_rank0(f"[relay] num_tokens[0] = {num_tokens[0].tolist()}")

            router = torch.stack([
                (torch.arange(num_tokens[i].shape[0] * 2, device=dev) % 2 == 0)
                .repeat_interleave(num_tokens[i].repeat_interleave(2))
                for i in range(bsz)
            ], dim=0)

            for i in range(bsz):
                half = concat_ids[i][router[i]]
                concat_ids[i][router[i]] = torch.where(
                    reveal[i], inputs["input_ids"][i], half)      # ground truth up to r
                keep_half[i] = keep_half[i] & ~reveal[i]          # drop them from the loss
                keep[i][router[i]] = keep_half[i]

            n_pred = int(keep_half.sum())
            p_mask = torch.ones(n_pred, device=dev, dtype=p_mask.dtype)
            bd = (concat_ids, concat_pos, attn, keep_half, keep, p_mask)

        # --- pass 2 (or the only pass, when relay is off) ---
        outputs = model(**inputs, output_hidden_states=True,
                        prev_latent=prev_latent, bd_inputs=bd)
        task_loss = outputs["loss"]

        # The model divides by all answer tokens, but we now predict fewer.
        # Without this, the relay arm's loss looks lower for a purely mechanical reason.
        if use_relay and n_pred:
            task_loss = task_loss * int((labels != -100).sum()) / n_pred


        # AR CE loss on clean (x0) region with Dream-shift-aligned labels.
        # hidden[i] predicts token[i+1], so shift labels by 1.
        shifted_labels = labels[:, 1 : min(seq_len + 1, labels.shape[1])].contiguous()
        if shifted_labels.shape[1] < seq_len:
            shifted_labels = F.pad(shifted_labels, (0, seq_len - shifted_labels.shape[1]), value=-100)

        _clean_h = outputs["hidden_states"][-1][:, seq_len : seq_len + seq_len, :]
        clean_logits = unwrapped_model.lm_head(
            _clean_h.to(unwrapped_model.lm_head.weight.dtype)
        )
        ce_logits = clean_logits.view(-1, clean_logits.size(-1))
        ce_labels = shifted_labels.view(-1)
        clean_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(ce_logits, ce_labels)

        if self.finetuning_args.loss_auto_balance:
            scale = task_loss.detach() / (clean_ce_loss.detach() + 1e-8)
            combined_loss = task_loss + scale * clean_ce_loss
        else:
            combined_loss = task_loss + alpha * clean_ce_loss

        # Logging
        log_dict = {
            "train/task_loss": task_loss.item(),
            "train/clean_ce_loss": clean_ce_loss.item(),
            "train/combined_loss": combined_loss.item(),
            "train/alpha": alpha,
        }
        if use_relay:
            ln = [p for n, p in unwrapped_model.named_parameters()
                  if "layer_norm" in n and p.requires_grad]
            if ln:
                log_dict["train/ln_weight_norm"] = ln[0].norm().item()
        self.log(log_dict)


        if self.state.global_step % self.args.logging_steps == 0:
            logger.info_rank0(
                f"Step {self.state.global_step}: "
                f"task_loss={task_loss.item():.4f}, "
                f"clean_ce_loss={clean_ce_loss.item():.4f}, "
                f"combined={combined_loss.item():.4f}"
            )

        return combined_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
