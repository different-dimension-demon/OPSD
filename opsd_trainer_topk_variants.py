import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from trl.trainer.utils import empty_cache

from opsd_trainer import OPSDTrainer


class _TopKAdvantageTrainerBase(OPSDTrainer):
    """Shared on-policy top-k loss helpers for experimental OPSD variants."""

    def _make_minimal_output(self):
        class MinimalOutput:
            def __init__(self):
                self.loss = None

        return MinimalOutput()

    def _teacher_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _sampled_log_probs(self, logits, sampled_token_ids, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        sampled_log_probs = torch.gather(
            log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs
        return sampled_log_probs

    def _teacher_top_k_forward_kl_per_token(self, student_logits, teacher_logits, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        kl_terms = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=0,
            temperature=temperature,
            reduction="none",
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        per_token_kl = kl_terms.sum(dim=-1)
        if self.top_k_loss is not None and self.top_k_loss > 0:
            per_token_kl = per_token_kl / self.top_k_loss
        return per_token_kl

    def _top_k_jsd_per_token(self, student_logits, teacher_logits):
        jsd_terms = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=self.beta,
            temperature=self.temperature,
            reduction="none",
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        return jsd_terms.sum(dim=-1)

    @staticmethod
    def _valid_token_mask(shifted_labels):
        if shifted_labels is None:
            return None
        return shifted_labels != -100

    @staticmethod
    def _mean_over_valid_tokens(per_token_loss, valid_mask):
        if valid_mask is None:
            return per_token_loss.mean()
        return per_token_loss.masked_fill(~valid_mask, 0).sum() / valid_mask.sum().clamp_min(1)

    def _student_teacher_tensors(self, model, inputs, log_prob_temperature=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_sampled_log_probs = self._sampled_log_probs(
            student_logits,
            sampled_token_ids,
            temperature=log_prob_temperature,
        )
        del outputs_student
        empty_cache()

        with torch.no_grad(), self._teacher_context(model):
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            teacher_sampled_log_probs = self._sampled_log_probs(
                teacher_logits,
                sampled_token_ids,
                temperature=log_prob_temperature,
            )
            del outputs_teacher
            empty_cache()

        advantage = (teacher_sampled_log_probs - student_sampled_log_probs).detach()
        return (
            student_logits,
            teacher_logits,
            student_sampled_log_probs,
            teacher_sampled_log_probs,
            advantage,
            shifted_labels,
        )


class TopKDropNegativePositionOPSDTrainer(_TopKAdvantageTrainerBase):
    """Top-k OPSD loss with negative sampled-token advantage positions zeroed."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        minimal_output = self._make_minimal_output() if return_outputs else None
        (
            student_logits,
            teacher_logits,
            student_sampled_log_probs,
            teacher_sampled_log_probs,
            advantage,
            shifted_labels,
        ) = self._student_teacher_tensors(model, inputs)

        per_token_loss = self._top_k_jsd_per_token(student_logits, teacher_logits)
        nonnegative_mask = advantage >= 0
        valid_mask = self._valid_token_mask(shifted_labels)
        loss_mask = nonnegative_mask if valid_mask is None else nonnegative_mask & valid_mask
        loss = self._mean_over_valid_tokens(per_token_loss, loss_mask)

        del (
            student_logits,
            teacher_logits,
            student_sampled_log_probs,
            teacher_sampled_log_probs,
            advantage,
            shifted_labels,
            per_token_loss,
            nonnegative_mask,
            valid_mask,
            loss_mask,
        )
        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss


class TopKAOPDNonPositiveOPSDTrainer(_TopKAdvantageTrainerBase):
    """AOPD non-positive handling with top-k teacher forward-KL guidance."""

    @staticmethod
    def _aopd_loss_temperature():
        return float(os.environ.get("OPSD_AOPD_LOSS_TEMPERATURE", "1.0"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        minimal_output = self._make_minimal_output() if return_outputs else None
        loss_temperature = self._aopd_loss_temperature()
        (
            student_logits,
            teacher_logits,
            student_sampled_log_probs,
            teacher_sampled_log_probs,
            advantage,
            shifted_labels,
        ) = self._student_teacher_tensors(
            model,
            inputs,
            log_prob_temperature=loss_temperature,
        )

        positive_mask = advantage > 0
        nonpositive_mask = ~positive_mask

        positive_opd_loss = -(advantage * student_sampled_log_probs)
        positive_opd_loss = positive_opd_loss.masked_fill(~positive_mask, 0)

        teacher_guidance_loss = self._teacher_top_k_forward_kl_per_token(
            student_logits,
            teacher_logits,
            temperature=loss_temperature,
        )
        teacher_guidance_loss = teacher_guidance_loss.masked_fill(~nonpositive_mask, 0)

        per_token_loss = positive_opd_loss + teacher_guidance_loss
        loss = self._mean_over_valid_tokens(per_token_loss, self._valid_token_mask(shifted_labels))

        del (
            student_logits,
            teacher_logits,
            student_sampled_log_probs,
            teacher_sampled_log_probs,
            advantage,
            shifted_labels,
            positive_opd_loss,
            teacher_guidance_loss,
            per_token_loss,
        )
        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss
