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
        self.normalize_wav2vec = kwargs.pop('normalize_wav2vec', True)
        self.normalize_type = kwargs.pop('normalize_type', 'batch')
        
        assert self.time_pooling_size > 0, "time_pooling_size must be greater than 0"
        assert self.normalize_type in ['batch', 'layer'], "normalize_type must be either 'batch' or 'layer'"

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

        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

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
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.pooling(hidden_states)

        if self.config.normalize_wav2vec:
            if self.config.normalize_type == 'batch':
                hidden_states = hidden_states.transpose(1, 2)
                hidden_states = self.norm(hidden_states)
                hidden_states = hidden_states.transpose(1, 2)
            else:
                hidden_states = F.layer_norm()

        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            if self.config.time_pooling_size > 1:
                input_lengths = input_lengths // self.config.time_pooling_size

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
