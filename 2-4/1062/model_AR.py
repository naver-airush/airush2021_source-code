#%%
import math
import copy

from typing import Optional, Any

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm

from torch.nn.init import xavier_uniform_


#%%
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_size) # seq length, hidden size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


#%%
class TransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, gpt_emb: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, gpt_emb, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, gpt_emb: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, gpt_emb, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, gpt_emb: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2_2 = self.self_attn(src, gpt_emb, gpt_emb, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        
        src2 = src2*0.5 + src2_2*0.5
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, gpt_emb: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # print(tgt.shape, tgt_mask.shape) # torch.Size([81, 128, 512]) torch.Size([81, 81])

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_2 = self.self_attn(tgt, gpt_emb, gpt_emb, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]        
        tgt2 = tgt2*0.5 + tgt2_2*0.5

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_encoder_layers, num_decoder_layers, intermediate_size, dropout=0.1):
        super(TransformerModel, self).__init__()

        encoder_layer = TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout, activation='relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout, activation='relu')
        decoder_norm = LayerNorm(hidden_size)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = hidden_size
        self.nhead = num_attention_heads

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.gpt_embeddings = nn.Embedding(vocab_size, hidden_size)        
        
        self.position_embeddings = PositionalEncoding(hidden_size, max_len=5000)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.decoder_embeddings = nn.Linear(hidden_size, vocab_size) # hidden size => vocab size
        self.decoder_embeddings.weight = self.token_embeddings.weight
        self.init_weights()

        # for annotation classification
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels = hidden_size,
                      out_channels = 128, # kernel_depth
                      kernel_size = i)
            for i in range(3,6) # 3,4,5
        ])
        self.fc = nn.Linear(3*128, 8)
        self.dropout2 = nn.Dropout(p=0.5)
        
    def set_gpt_emb(self, emb, freeze) :
        self.gpt_embeddings = emb
        if freeze :
            for emb in self.gpt_embeddings.parameters() :
                emb.requires_grad = False

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder_embeddings.bias.data.zero_()
        self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            src_embeddings = self.token_embeddings(src) * math.sqrt(self.hidden_size) + self.position_embeddings(src)
            src_embeddings = self.dropout(src_embeddings)
            
            src_gpt_embeddings = self.gpt_embeddings(src)

            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.t()

            if tgt is None:  # encode
                memory = self.encoder(src_embeddings, src_gpt_embeddings, src_key_padding_mask=src_key_padding_mask)
                return memory

        if tgt is not None:
            tgt_embeddings = self.token_embeddings(tgt) * math.sqrt(self.hidden_size) + self.position_embeddings(tgt)
            tgt_embeddings = self.dropout(tgt_embeddings)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

            tgt_gpt_embeddings = self.gpt_embeddings(tgt)

            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.t()

            if src is None and memory is not None:  # decode
                if memory_key_padding_mask is not None:
                    memory_key_padding_mask = memory_key_padding_mask.t()

                output = self.decoder(tgt_embeddings, tgt_gpt_embeddings, memory,
                                      tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
                output = self.decoder_embeddings(output)
                return output

        
        # training
        assert not (src is None and tgt is None)

        memory = self.encoder(src_embeddings, src_gpt_embeddings, src_key_padding_mask=src_key_padding_mask)

        x_convs = [F.relu(conv1d(memory.permute(1, 2, 0))) for conv1d in self.convs]
        x_pools = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_convs]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pools], dim=1)
        memory_conv = self.fc(self.dropout2(x_fc))

        output = self.decoder(tgt_embeddings, tgt_gpt_embeddings, memory, 
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.decoder_embeddings(output)

        return memory_conv, output



#%%
class TransformerDecoder2(nn.Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder2, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # self_attn_probs = []
        for mod in self.layers:
            tgt = mod(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            # self_attn_probs.append(self_attn_prob)
        return tgt


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_hidn):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=d_hidn, out_channels=d_hidn * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_hidn * 4, out_channels=d_hidn, kernel_size=1)
        self.active = F.gelu

    def forward(self, inputs):
        output = self.active(self.conv1(inputs.permute(1,2,0)))
        output = self.conv2(output).permute(2,0,1)
        return output


class TransformerDecoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer2, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.pos_ffn  = PoswiseFeedForwardNet(d_model)
        self.norm3 = LayerNorm(d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer2, self).__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_outputs, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                             key_padding_mask=tgt_key_padding_mask)
        self_att_outputs = self.norm1(tgt+self_att_outputs)
        ffn_outputs = self.pos_ffn(self_att_outputs)
        ffn_outputs = self.norm3(self_att_outputs + ffn_outputs)
        return ffn_outputs


class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_decoder_layers, intermediate_size, dropout):
        super().__init__()
        decoder_layer = TransformerDecoderLayer2(hidden_size, num_attention_heads, intermediate_size, dropout, activation='relu')
        decoder_norm = LayerNorm(hidden_size)
        self.decoder = TransformerDecoder2(decoder_layer, num_decoder_layers, decoder_norm)

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEncoding(hidden_size, max_len=5000)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, dec_inputs):

        # if dec_self_attn_mask is not None:
        #     dec_self_attn_mask = dec_self_attn_mask.t()
        
        tgt_mask = self.generate_square_subsequent_mask(dec_inputs.size(0)).to(dec_inputs.device)
        dec_embeddings = self.token_embeddings(dec_inputs) * math.sqrt(self.hidden_size) + self.position_embeddings(dec_inputs)
        # 여기까지는 잘 나오는 것 같음
        
        dec_outputs = self.decoder(dec_embeddings, tgt_mask, tgt_key_padding_mask=None)
        # print(dec_outputs.shape) # length, batch, hidden
        return dec_outputs


class GPTPretrain(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_decoder_layers, intermediate_size, dropout):
        super().__init__()

        self.gpt = GPT(vocab_size, hidden_size, num_attention_heads, num_decoder_layers, intermediate_size, dropout)
        self.projection_lm = nn.Linear(hidden_size, vocab_size, bias=False)
        self.projection_lm.weight = self.gpt.token_embeddings.weight
    
    def forward(self, dec_inputs):
        dec_outputs = self.gpt(dec_inputs)
        logits_lm = self.projection_lm(dec_outputs)
        logits_lm = logits_lm.permute(1,0,2) # batch, length, vocab
        return logits_lm.contiguous()



