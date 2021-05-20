#!/usr/bin/env python3
import torch
from torch import nn

import transformers
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Config
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.activations import ACT2FN
import torch.nn.functional as F
import copy

class Pooling1d(nn.Module):
    def __init__(
        self,
        pool_type,
        kernel_size,
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__()
        self.time_pooling_size = kernel_size

        if stride is None:
            stride = kernel_size

        if pool_type == "avg":
            self.pool_layer = torch.nn.AvgPool1d(
                kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            )

        elif pool_type == "max":
            self.pool_layer = torch.nn.MaxPool1d(
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

    def forward(self, x):
        if self.time_pooling_size == 1:
            return x

        # Put the pooling axes as the last dimension for torch.nn.pool
        x = x.transpose(-1, 1)

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        x = x.transpose(-1, 1)

        return x


class Wav2Vec2Config(Wav2Vec2Config):
    def __init__(
        self,
        **kwargs
    ):
        self.time_pooling_size = kwargs.pop('time_pooling_size', 1)
        self.pooling_type = kwargs.pop('pooling_type', None)
        self.vocab_size_char = kwargs.pop('vocab_size_char', None)
        self.vocab_size_bpe = kwargs.pop('vocab_size_bpe', None)

        assert self.time_pooling_size > 0, "time_pooling_size must be greater than 0"
        assert self.vocab_size_char is None, "vocab_size_char must be set"
        assert self.vocab_size_bpe is None, "vocab_size_bpe must be set"

        super().__init__(**kwargs)


class Wav2Vec2ForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2Config
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        self.pooling = Pooling1d(
            pool_type=config.pooling_type,
            kernel_size=config.time_pooling_size,
        )

        self.norm = nn.BatchNorm1d(
            config.hidden_size,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        self.dropout_char = nn.Dropout(config.final_dropout)
        self.lm_head_char = nn.Linear(config.hidden_size, config.vocab_size_char)

        self.dropout_bpe = nn.Dropout(config.final_dropout)
        self.lm_head_bpe = nn.Linear(config.hidden_size, config.vocab_size_bpe)

        self.char_weight = 0.3
        
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels_bpe=None,
        labels_char=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x = outputs[0]


        hidden_states = self.dropout_char(x)
        logits_char = self.lm_head_char(hidden_states)


        hidden_states = self.pooling(x)
        hidden_states = self.dropout_bpe(hidden_states)
        logits_bpe = self.lm_head_bpe(hidden_states)



        loss = None
        if labels_char is not None:

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths_char = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_char_mask = labels_char >= 0
            target_char_lengths = labels_char_mask.sum(-1)
            flattened_targets_char = labels_char.masked_select(labels_char_mask)

            log_probs_char = F.log_softmax(logits_char, dim=-1).transpose(0, 1)

            loss_char = None
            with torch.backends.cudnn.flags(enabled=False):
                loss_char = F.ctc_loss(
                    log_probs_char,
                    flattened_targets_char,
                    input_lengths_char,
                    target_char_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )


            input_lengths_bpe = copy.deepcopy(input_lengths_char)
            if self.config.time_pooling_size > 1:
                input_lengths_bpe = input_lengths_bpe // self.config.time_pooling_size

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_bpe_mask = labels_bpe >= 0
            target_bpe_lengths = labels_bpe_mask.sum(-1)
            flattened_targets_bpe = labels_bpe.masked_select(labels_bpe_mask)

            log_probs_bpe = F.log_softmax(logits_bpe, dim=-1).transpose(0, 1)

            loss_bpe = None
            with torch.backends.cudnn.flags(enabled=False):
                loss_bpe = F.ctc_loss(
                    log_probs_bpe,
                    flattened_targets_bpe,
                    input_lengths_bpe,
                    target_bpe_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            
            loss = self.char_weight * loss_char
            loss += (1 - self.char_weight) * loss_bpe


        if not return_dict:
            output = (logits_bpe,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits_bpe, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
