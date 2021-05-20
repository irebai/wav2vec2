
# speech brain functions
from speechbrain.lobes.models.VanillaNN import VanillaNN
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.RNN import AttentionalRNNDecoder
from speechbrain.nnet.losses import nll_loss

class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config, layer_id=0, size=512):
        super().__init__()
        if layer_id == 0:
            self.intermediate_dense = nn.Linear(config.hidden_size, size)
        else:
            self.intermediate_dense = nn.Linear(size, size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class Wav2Vec2Model_freez(Wav2Vec2Model):
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

class Encoder(nn.Module):
    def __init__(self, config, n_layers=2, size=512):
        super().__init__()
    
        ff = [
            Wav2Vec2FeedForward(config, layer_id=i, size=size) for i in range(n_layers)
        ]
        self.ff = nn.ModuleList(ff)
    def forward(self, hidden_states):
        for f_layer in self.ff:
            hidden_states = f_layer(hidden_states)
        return hidden_states

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dec_neurons = 256
        self.emb_dim = 128
        self.enc_dim = 1024
        
        self.enc = VanillaNN(
            input_shape=[None, None, self.enc_dim],
            activation=torch.nn.LeakyReLU,
            dnn_blocks=2,
            dnn_neurons=self.enc_dim
        )
        self.emb = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=self.emb_dim
        )
        self.dec = AttentionalRNNDecoder(
            enc_dim=self.enc_dim,
            input_size=self.emb_dim,
            rnn_type='gru',
            attn_type='location',
            hidden_size=self.dec_neurons,
            attn_dim=256,
            num_layers=1,
            scaling=1.0,
            channels=10,
            kernel_size=100,
            re_init=True,
            dropout=0.5
        )
        
        self.dropout = nn.Dropout(config.final_dropout)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.seq_head = nn.Linear(self.dec_neurons, config.vocab_size)
        
        
    def forward(self, x, labels_bos, wav_lens):
        ctc_logits = None
        seq_logits = None
        # output layer for ctc
        #x = self.enc(x)
        ctc_logits = self.ctc_head(self.dropout(x))
    
        # output layer for seq2seq
        #e_in = self.emb(labels_bos)
        #h, _ = self.dec(e_in, x, wav_lens)
        #seq_logits = self.seq_head(self.dropout(h))
        
        return ctc_logits, seq_logits
    
    
class EncoderDecoder(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.ctc_weight = 0.2
        
        self.wav2vec2 = Wav2Vec2Model_freez(config)
        self.model = Model(config)
        
        self.init_weights()

    def freeze_model(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2._freeze_parameters()
        
    def test(self):
        for param in self.parameters():
            print(param.requires_grad)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # add bos token to labels
        if labels is not None:
            labels_mask = labels >= 0
            labels_lens = labels_mask.sum(-1)
            labels[labels == -100] = 0
            
            #BOS
            labels_bos_mask = labels_bos >= 0
            labels_bos_lens = labels_bos_mask.sum(-1)
            labels_bos[labels_bos == -100] = 0
            
            #EOS
            labels_eos_mask = labels_eos >= 0
            labels_eos_lens = labels_eos_mask.sum(-1)
            labels_eos[labels_eos == -100] = 0
        
        # retrieve input_lengths from attention_mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        feat_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        wav_lens = attention_mask.sum(-1).type(torch.int32)
        wav_lens = wav_lens / torch.max(wav_lens)
        
        
        
        # compute forward
        feat = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        feat = feat[0]
        ctc_logits, seq_logits = self.model(feat, labels_bos, wav_lens)
        
        # output prob
        p_ctc = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        #p_seq = F.log_softmax(seq_logits, dim=-1)
        
        
        loss = None
        if labels is not None:
            loss = ctc_loss = self.compute_ctc_loss(p_ctc, labels, feat_lengths, labels_lens)
            #seq_loss = nll_loss(p_seq, labels_eos, labels_eos_lens, label_smoothing=0.1, reduction=self.config.ctc_loss_reduction)
            #seq_loss = self.compute_nll_loss(p_seq, labels_eos, labels_eos_mask, label_smoothing=0.1)
            
            
            #loss = self.ctc_weight * ctc_loss
            #loss += (1 - self.ctc_weight) * seq_loss
            
 
        if not return_dict:
            output = (seq_logits,) + outputs[1:]
            return ((seq_logits,) + output) if loss is not None else output

        
        return CausalLMOutput(
            loss=loss, logits=ctc_logits, hidden_states=None, attentions=None
        )

    def compute_ctc_loss(self, log_probs, labels, input_lengths, labels_lens):
        loss = None
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                labels_lens,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
        return loss
    
    def compute_nll_loss(self, log_probs, labels, mask, label_smoothing=0):
        # Compute, then reduce loss
        with torch.backends.cudnn.flags(enabled=False):
            loss = torch.nn.functional.nll_loss(
                log_probs,
                labels,
            )
        
        reduction=self.config.ctc_loss_reduction

        N = loss.size(0)
        if reduction == "mean":
            loss = loss.sum() / torch.sum(mask)
        elif reduction == "batchmean":
            loss = loss.sum() / N
        elif reduction == "batch":
            loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

        if label_smoothing == 0:
            return loss
        else:
            loss_reg = torch.mean(log_probs, dim=1) * mask
            if reduction == "mean":
                loss_reg = torch.sum(loss_reg) / torch.sum(mask)
            elif reduction == "batchmean":
                loss_reg = torch.sum(loss_reg) / labels.shape[0]
            elif reduction == "batch":
                loss_reg = loss_reg.sum(1) / mask.sum(1)

            return -label_smoothing * loss_reg + (1 - label_smoothing) * loss

                
