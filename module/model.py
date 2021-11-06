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
        dim=None,
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__()
        self.time_pooling_size = kernel_size

        assert pool_type in ['avg', 'max', 'conv', 'cat'], "pool_type not in ['avg', 'max', 'conv', 'cat']"
        self.pool_type = pool_type

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
        elif pool_type == "conv":
            assert dim is not None,"you have to precise the dimension of the ConvLayer"
            self.conv = nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
            )
            self.act = ACT2FN["gelu"]

    def forward(self, x):
        if self.time_pooling_size == 1:
            return x
        
        if self.pool_type == 'cat':
            batch, frames, features = x.shape
            if frames % self.time_pooling_size > 0:
                tzero = torch.zeros(batch, frames % self.time_pooling_size, features).to('cuda')
                x = torch.cat((x, tzero), dim=1)
                batch, frames, features = x.shape

            x = torch.reshape(x, [batch, int(frames / self.time_pooling_size), int(features * self.time_pooling_size)])

            return x
        
        elif self.pool_type == 'conv':
            x = x.transpose(-1, 1)
            x = self.conv(x)
            x = x.transpose(-1, 1)
            x = self.act(x)
            return x
        
        else:
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
        self.time_pooling_size = 1
        self.pooling_type = 'max'
        self.normalize_wav2vec = False
        self.normalize_type = 'batch'
        self.num_ff_layers = 0
        self.reduce_ff_layer = 1
        super().__init__(**kwargs)


class FeedForward(nn.Module):
    def __init__(self,
        ffn_in,
        ffn_out,
        activation_dropout,
    ):
        super().__init__()
        self.drop = nn.Dropout(activation_dropout)
        self.ffn = nn.Linear(ffn_in, ffn_out)
        self.act = nn.LeakyReLU()

    def forward(self, hidden_states):
        hidden_states = self.drop(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.act(hidden_states)
        return hidden_states

class Wav2Vec2ForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2Config
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.wav2vec2 = Wav2Vec2Model(config)
        
        if config.pooling_type == 'cat':
            self.ffn_in = config.hidden_size * config.time_pooling_size
        else:
            self.ffn_in = config.hidden_size

        if config.normalize_wav2vec:
            if self.config.normalize_type == 'batch':
                self.norm = nn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
            else:
                self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.pooling = Pooling1d(
            pool_type=config.pooling_type,
            kernel_size=config.time_pooling_size,
            dim=config.hidden_size,
        )
        
        ff_layers_size = [int(config.hidden_size/(1 if i == 0 else config.reduce_ff_layer * i)) for i in range(config.num_ff_layers + 1)]
        self.ff_layers = nn.ModuleList(
            [
                FeedForward(
                    ff_layers_size[i],
                    ff_layers_size[i+1],
                    config.activation_dropout
                ) for i in range(config.num_ff_layers)
            ]
        )
        
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(ff_layers_size[-1], config.vocab_size)

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
        
        # normalize the output if required
        if self.config.normalize_wav2vec:
            if self.config.normalize_type == 'batch':
                hidden_states = hidden_states.transpose(1, 2)
                hidden_states = self.norm(hidden_states)
                hidden_states = hidden_states.transpose(1, 2)
            else:
                hidden_states = F.norm(hidden_states, hidden_states.shape)
                
        hidden_states = self.pooling(hidden_states)
        
        for ff_layer in self.ff_layers:
            hidden_states = ff_layer(hidden_states)
        
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
