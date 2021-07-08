#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Config
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.activations import ACT2FN

class Wav2Vec2Config(Wav2Vec2Config):
    def __init__(self, **kwargs):
        self.time_pooling_size = 1
        self.pooling_type = "max"
        self.normalize_wav2vec = False
        self.normalize_type = "layer"
        self.ctc_weight = 0.2
        self.embedding_dim = 128
        self.num_decoder_layers = 4
        super().__init__(**kwargs)
        
        
class Wav2Vec2ForS2S(Wav2Vec2ForCTC):
    config_class = Wav2Vec2Config
    def __init__(self, config):
        super().__init__(config)
        
        self.wav2vec2 = Wav2Vec2Model(config)
        
        if config.normalize_wav2vec:
            if self.config.normalize_type == 'batch':
                self.enc_norm = nn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
            else:
                self.enc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            

        self.dec = Decoder(config)

        self.ctc_dropout = nn.Dropout(config.final_dropout)
        self.ctc_lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.seq_dropout = nn.Dropout(config.final_dropout)
        self.seq_lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def get_lookahead_mask(self, padded_input):
        seq_len = padded_input.shape[1]
        mask = (
            torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
            == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.detach().to(padded_input.device)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        labels_bos=None,
        labels_eos=None,
    ):
        # retrieve loss input_lengths from attention_mask
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        input_lens = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

        if labels is not None and labels_bos is not None and labels_eos is not None:
            #ORIGINAL TARGET
            labels_mask = labels >= 0
            labels_lens = labels_mask.sum(-1)
            
            #ADD BOS TO TARGET
            labels_bos_mask = labels_bos < 0
            labels_bos_lens = labels_bos_mask.sum(-1)
            labels_bos[labels_bos == -100] = 0
            
            #ADD EOS TO TARGET
            labels_eos_mask = labels_eos >= 0
            labels_eos_lens = labels_eos_mask.sum(-1)

        
        # compute forward
        feat = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        
        # get wav2vec output
        encoder_out = feat[0]
        
        # normalize the output if required
        if self.config.normalize_wav2vec:
            if self.config.normalize_type == 'batch':
                encoder_out = encoder_out.transpose(1, 2)
                encoder_out = self.norm(encoder_out)
                encoder_out = encoder_out.transpose(1, 2)
            else:
                encoder_out = F.norm(encoder_out, hidden_states.shape)

        # output layer for ctc log-probabilities
        logits = self.ctc_lm_head(self.ctc_dropout(encoder_out))
        p_ctc = F.log_softmax(logits, dim=-1)

        decoder_out = self.dec(
            tgt=labels_bos,
            memory=encoder_out,
            tgt_mask=self.get_lookahead_mask(labels_bos),
            tgt_key_padding_mask=labels_bos_mask
        )
        
        # output layer for seq2seq log-probabilities
        logits = self.seq_lm_head(self.seq_dropout(decoder_out))
        p_seq = F.log_softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            with torch.backends.cudnn.flags(enabled=False):
                loss_ctc = F.ctc_loss(
                    p_ctc.transpose(0, 1),
                    labels.masked_select(labels_mask),
                    input_lens,
                    labels_lens,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                loss_seq = nn.functional.nll_loss(
                    p_seq.transpose(1, 2),
                    labels_eos,
                    ignore_index=-100,
                    reduce=None,
                    reduction=self.config.ctc_loss_reduction
                )

                loss = (
                    self.config.ctc_weight * loss_ctc
                    + (1 - self.config.ctc_weight) * loss_seq
                )

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=feat.hidden_states, attentions=feat.attentions
        )
        
class Decoder(nn.Module):
    def __init__(
        self,
        config,
        normalize_before=False,
    ):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.embedding_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)
        
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config,
                    normalize_before=normalize_before,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        output_attentions=False,
    ):
        """
        Arguments
        ----------
        tgt : tensor
            The sequence to the decoder layer (required).
        memory : tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : tensor
            The mask for the tgt sequence (optional).
        memory_mask : tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : tensor
            The mask for the memory keys per batch (optional).
        """
        hidden_states = self.emb(tgt.long())
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        output = hidden_states
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        if output_attentions:
            return output, (self_attns, multihead_attns)
        else:
            return output

class DecoderLayer(nn.Module):
    def __init__(self, config, normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None
        )
        self.mutihead_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None
        )
        self.pos_ffn = Wav2Vec2FeedForward(config)        
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # normalization layers
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        batch_first=True,
    ):
        #self attention
        if batch_first:
            tgt1 = tgt.transpose(1, 0)
        tgt1, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        if batch_first:
            tgt1 = tgt1.transpose(1, 0)
            
        #dropout, residual connection and layer norm
        tgt = self.norm1(tgt + self.dropout(tgt1))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        if batch_first:
            tgt1 = tgt.transpose(1, 0)
            memory = memory.transpose(1, 0)
        tgt1, multihead_attention = self.mutihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        if batch_first:
            tgt1 = tgt1.transpose(1, 0)
        
        #dropout, residual connection and layer norm
        tgt = self.norm2(tgt + self.dropout(tgt1))

        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        tgt1 = self.pos_ffn(tgt)
        
        #dropout, residual and layer norm
        tgt = self.norm3(tgt + self.dropout(tgt1))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return tgt, self_attn, multihead_attention
        
        
        
        
        
        
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt
        
        # self-attention over the target sequence
        if batch_first:
            tgt1 = tgt1.transpose(1, 0)
        tgt2, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        if batch_first:
            tgt2 = tgt2.transpose(1, 0)

        # add & norm
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states
        if batch_first:
            tgt1 = tgt1.transpose(1, 0)
            memory = memory.transpose(1, 0)
        
        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        
        if batch_first:
            tgt2 = tgt2.transpose(1, 0)
        
        
        # add & norm
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, multihead_attention
        """
    

class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
