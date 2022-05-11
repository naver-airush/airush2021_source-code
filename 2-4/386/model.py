from typing import Tuple
import argparse

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, BertConfig, EncoderDecoderConfig, EncoderDecoderModel, GPT2Config
from transformers import XLMRobertaConfig, XLMRobertaModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

import larva
from larva import LarvaModel

from fairseq import utils
from fairseq.modules import (
    gelu,
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


from custom_fairseq import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_encoder_layers, num_decoder_layers, intermediate_size, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEncoding(hidden_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.transformer = Transformer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=intermediate_size,
            dropout=dropout,
        )

        self.decoder_embeddings = nn.Linear(hidden_size, vocab_size)
        self.decoder_embeddings.weight = self.token_embeddings.weight

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder_embeddings.bias.data.zero_()
        self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            src_embeddings = self.token_embeddings(src) * math.sqrt(self.hidden_size) + self.position_embeddings(src)
            src_embeddings = self.dropout(src_embeddings)

            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.t()

            if tgt is None:  # encode
                memory = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
                return memory

        if tgt is not None:
            tgt_embeddings = self.token_embeddings(tgt) * math.sqrt(self.hidden_size) + self.position_embeddings(tgt)
            tgt_embeddings = self.dropout(tgt_embeddings)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.t()

            if src is None and memory is not None:  # decode
                if memory_key_padding_mask is not None:
                    memory_key_padding_mask = memory_key_padding_mask.t()

                output = self.transformer.decoder(
                    tgt_embeddings, 
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                output = self.decoder_embeddings(output)
                return output

        assert not (src is None and tgt is None)
        output = self.transformer(src_embeddings,
            tgt_embeddings, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.decoder_embeddings(output)
        return output


    def forward1(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            src_embeddings = self.token_embeddings(src) * math.sqrt(self.hidden_size) + self.position_embeddings(src)
            src_embeddings = self.dropout(src_embeddings)
            print(f"source embedding dimension is {src_embeddings.shape}")
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.t()
                print(f"source key padding mask dimension is {src_key_padding_mask.shape}")

            if tgt is None:  # encode
                memory = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
                print(f"memory dimension is {memory.shape}")
                return memory

        if tgt is not None:
            tgt_embeddings = self.token_embeddings(tgt) * math.sqrt(self.hidden_size) + self.position_embeddings(tgt)
            tgt_embeddings = self.dropout(tgt_embeddings)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            print(f"target embedding dimension is {tgt_embeddings.shape}")
            print(f"target mask dimension is {tgt_mask.shape}")

            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.t()
                print(f"target key padding mask dimension is {tgt_key_padding_mask.shape}")

            if src is None and memory is not None:  # decode
                if memory_key_padding_mask is not None:
                    memory_key_padding_mask = memory_key_padding_mask.t()
                    print(f"memory key padding mask dimension is {memory_key_padding_mask.shape}")

                output = self.transformer.decoder(
                    tgt_embeddings, 
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                print(f"output dimension is {output.shape}")
                output = self.decoder_embeddings(output)
                print(f"output dimension is {output.shape}")
                return output

        assert not (src is None and tgt is None)
        output = self.transformer(src_embeddings,
            tgt_embeddings, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        print(f"output dimension is {output.shape}")
        output = self.decoder_embeddings(output)
        print(f"output dimension is {output.shape}")
        return output



class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_encoder_layers, num_decoder_layers, intermediate_size, dropout=0.1):
        super(EncoderDecoder, self).__init__()

        self.config_encoder = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_encoder_layers,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            vocab_size=vocab_size,
            pad_token_id=1,
        )

        self.config_decoder = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_decoder_layers,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            vocab_size=vocab_size,
            pad_token_id=1,
        )

        # set decoder config to causal lm
        self.config_decoder.is_decoder = True
        self.config_decoder.add_cross_attention = True


        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.config_encoder, self.config_decoder)

        # Initializing a Bert2Bert model from the bert-base-uncased style configurations
        self.model = EncoderDecoderModel(config=self.config)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        # Accessing the model configuration
        self.config_encoder = self.model.config.encoder
        self.config_decoder  = self.model.config.decoder

        self.mlm_head = BertOnlyMLMHead(self.config_encoder)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        pass
        # self.model.decoder.bert.embeddings.word_embeddings.weight = self.model.encoder.embeddings.word_embeddings.weight
        # initrange = 0.1
        # self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.decoder_embeddings.bias.data.zero_()
        # self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            outputs = self.model(
                input_ids=src,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )
        if memory is not None:
            outputs = self.model(
                encoder_outputs=memory,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )

        return outputs.logits
    
    @staticmethod
    def mask_tokens(
        inputs : torch.Tensor, 
        tokenizer : BertTokenizer, 
        config : argparse.Namespace, 
        mlm_probability : float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Args:
            inputs (torch.Tensor)
            tokenizer (BertTokenizer)
            config (argparse.Namespace)
            mlm_probability (float, optional). Defaults to 0.15.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input_id, labels
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
        #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        probability_matrix.masked_fill_(torch.eq(labels, tokenizer.pad_token_id), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(device=inputs.device, dtype=torch.bool) & masked_indices
        # inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids("<mask>")

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(device=inputs.device, dtype=torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(config.vocab_size, labels.shape, device=inputs.device, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].to(inputs.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def forward_pretrain(
        self : nn.Module, 
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        tokenizer : BertTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pretrain forward
        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            tokenizer (BertTokenizer)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logits, labels
        """
        input_ids, labels = self.mask_tokens(input_ids, tokenizer, self.config_encoder)
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mlm_logits = self.mlm_head(outputs.last_hidden_state)
        
        return mlm_logits, labels



    def forward1(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            outputs = self.model(
                input_ids=src,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )
        if memory is not None:
            outputs = self.model(
                encoder_outputs=memory,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )

        return outputs.logits


class BertGpt2(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_encoder_layers, num_decoder_layers, intermediate_size, dropout=0.1):
        super(BertGpt2, self).__init__()

        self.config_encoder = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_encoder_layers,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            vocab_size=vocab_size,
            pad_token_id=1,
        )

        self.config_decoder = GPT2Config(
            n_embd=hidden_size,
            n_inner=intermediate_size,
            n_head=num_attention_heads,
            n_layer=num_decoder_layers,
            attn_pdrop=dropout,
            embd_pdrop=dropout,
            vocab_size=vocab_size,
        )

        # set decoder config to causal lm
        self.config_decoder.is_decoder = True
        self.config_decoder.add_cross_attention = True


        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.config_encoder, self.config_decoder)

        # Initializing a Bert2Bert model from the bert-base-uncased style configurations
        self.model = EncoderDecoderModel(config=self.config)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        # Accessing the model configuration
        self.config_encoder = self.model.config.encoder
        self.config_decoder  = self.model.config.decoder

        self.mlm_head = BertOnlyMLMHead(self.config_encoder)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.model.decoder.transformer.wte.weight = self.model.encoder.embeddings.word_embeddings.weight
        # model.decoder.transformer.wte.weight
        # self.model.decoder.bert.embeddings.word_embeddings.weight = self.model.encoder.embeddings.word_embeddings.weight
        # initrange = 0.1
        # self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.decoder_embeddings.bias.data.zero_()
        # self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            outputs = self.model(
                input_ids=src,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )
        if memory is not None:
            outputs = self.model(
                encoder_outputs=memory,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )

        return outputs.logits


    @staticmethod
    def mask_tokens(
        inputs : torch.Tensor, 
        tokenizer : BertTokenizer, 
        config : argparse.Namespace, 
        mlm_probability : float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Args:
            inputs (torch.Tensor)
            tokenizer (BertTokenizer)
            config (argparse.Namespace)
            mlm_probability (float, optional). Defaults to 0.15.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input_id, labels
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
        #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        probability_matrix.masked_fill_(torch.eq(labels, tokenizer.pad_token_id), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(device=inputs.device, dtype=torch.bool) & masked_indices
        # inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids("<mask>")

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(device=inputs.device, dtype=torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(config.vocab_size, labels.shape, device=inputs.device, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].to(inputs.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def forward_pretrain(
        self : nn.Module, 
        input_ids : torch.Tensor,
        tokenizer : BertTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pretrain forward
        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            tokenizer (BertTokenizer)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logits, labels
        """
        input_ids, labels = self.mask_tokens(input_ids, tokenizer, self.config_encoder)
        outputs = self.model.encoder(input_ids=input_ids)
        mlm_logits = self.mlm_head(outputs.last_hidden_state)
        
        return mlm_logits, labels


    def forward1(self, src=None, tgt=None, memory=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src is not None:
            outputs = self.model(
                input_ids=src,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )
        if memory is not None:
            outputs = self.model(
                encoder_outputs=memory,
                attention_mask=src_key_padding_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_key_padding_mask,
            )


        return outputs.logits


class TransformerS2Model(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, tokenizer, mask_cls_sep=False, args=None):
        super().__init__(tokenizer, mask_cls_sep, args)
        
        encoder_embed_tokens = self.build_embedding(args, tokenizer)
        self.encoder = TransformerS2Encoder(args, encoder_embed_tokens)
        
        decoder_embed_tokens = self.build_embedding(args, tokenizer)
        self.decoder = TransformerDecoder(args, decoder_embed_tokens)

        if args.tokenizer in ['char', 'pre_char', 'kfold_char', 'wordpiece']:
            if args.bert == 'original':
                self.config = BertConfig(
                    hidden_size=args.hidden_size,
                    # intermediate_size=args.bert_intermediate_size,
                    # num_attention_heads=args.bert_num_attention_heads,
                    # num_hidden_layers=args.num_hidden_layers,
                    intermediate_size=args.intermediate_size,
                    num_attention_heads=args.num_attention_heads,
                    num_hidden_layers=args.num_encoder_layers,
                    attention_probs_dropout_prob=args.dropout,
                    hidden_dropout_prob=args.dropout,
                    vocab_size=args.vocab_size,
                    pad_token_id=tokenizer.pad_token_id,
                )            
                self.bert_encoder = BertModel(self.config)
            elif args.bert == 'xlm':
                self.config = XLMRobertaConfig(
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                    num_attention_heads=args.num_attention_heads,
                    num_hidden_layers=args.num_encoder_layers,
                    attention_probs_dropout_prob=args.dropout,
                    hidden_dropout_prob=args.dropout,
                    vocab_size=args.vocab_size,
                    pad_token_id=tokenizer.pad_token_id,
                )            
                self.bert_encoder = XLMRobertaModel(self.config)
        elif args.tokenizer in ['plus']:
            self.bert_encoder = LarvaModel.from_pretrained("larva-kor-plus-base-cased")  # BertModel
            self.config = self.bert_encoder.config
            print(self.config)

        self.mlm_head = BertOnlyMLMHead(self.config)

    def build_embedding(self, args, tokenizer):
        num_embeddings = args.vocab_size
        padding_idx = tokenizer.pad_token_id
        emb = Embedding(num_embeddings, args.hidden_size, padding_idx)
        return emb    

    def forward(self, src_tokens, prev_output_tokens, bert_input, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        bert_encoder_padding_mask = bert_input.eq(self.config.pad_token_id)
        
        bert_encoder_out =  self.bert_encoder(bert_input, attention_mask=~bert_encoder_padding_mask)
        bert_encoder_out = bert_encoder_out.last_hidden_state
        
#         if self.mask_cls_sep:
#             bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.cls())
#             bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.sep())

        bert_encoder_out = bert_encoder_out.permute(1,0,2).contiguous()
        bert_encoder_out = {
            'bert_encoder_out': bert_encoder_out,
            'bert_encoder_padding_mask': bert_encoder_padding_mask,
        }
        
        encoder_out = self.encoder(src_tokens, bert_encoder_out=bert_encoder_out)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, bert_encoder_out=bert_encoder_out, **kwargs)
        return decoder_out[0]

    @staticmethod
    def mask_tokens(
        inputs : torch.Tensor, 
        tokenizer : BertTokenizer, 
        config : argparse.Namespace, 
        mlm_probability : float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Args:
            inputs (torch.Tensor)
            tokenizer (BertTokenizer)
            config (argparse.Namespace)
            mlm_probability (float, optional). Defaults to 0.15.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input_id, labels
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
        #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        probability_matrix.masked_fill_(torch.eq(labels, tokenizer.pad_token_id), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(device=inputs.device, dtype=torch.bool) & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        # inputs[indices_replaced] = tokenizer.convert_tokens_to_ids("<mask>")

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(device=inputs.device, dtype=torch.bool) & masked_indices & ~indices_replaced
        random_words = torch.randint(config.vocab_size, labels.shape, device=inputs.device, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].to(inputs.device)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def forward_pretrain(
        self : nn.Module, 
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        tokenizer : BertTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pretrain forward
        Args:
            self (nn.Module)
            input_ids (torch.Tensor)
            tokenizer (BertTokenizer)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logits, labels
        """
        input_ids, labels = self.mask_tokens(input_ids, tokenizer, self.config)
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        mlm_logits = self.mlm_head(outputs.last_hidden_state)
        
        return mlm_logits, labels


class TransformerS2Encoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_tokens):
        super().__init__()

        self.dropout = args.dropout
        
        embed_dim = args.hidden_size
        self.padding_idx = 1
        self.max_source_positions = args.hidden_size
        
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        
        self.embed_positions = PositionalEmbedding(
            args.vocab_size, embed_dim, self.padding_idx,
            learned=True,
        )
        
#         self.embedding = nn.Embedding(args.vocab_size, embed_dim, self.padding_idx)
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerS2EncoderLayer(args)
            for i in range(args.num_encoder_layers)
        ])

        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, bert_encoder_out):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, bert_encoder_out['bert_encoder_out'], bert_encoder_out['bert_encoder_padding_mask'])

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerS2EncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        
        self.embed_dim = args.hidden_size

        self.self_attn = MultiheadAttention(
            self.embed_dim, args.num_attention_heads,
            dropout=args.dropout, self_attention=True
        )

        bert_out_dim = args.hidden_size
        
        self.bert_attn = MultiheadAttention(
            self.embed_dim, args.num_attention_heads,
            kdim=bert_out_dim, vdim=bert_out_dim,
            dropout=args.dropout,
        )
        
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        
        self.dropout = args.dropout
        
        self.activation_fn = F.relu
        
        self.activation_dropout = args.dropout
                
        self.fc1 = Linear(self.embed_dim, args.ffn_size)
        self.fc2 = Linear(args.ffn_size, self.embed_dim)
        
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_bert_dropout = args.dropout
        self.encoder_bert_dropout_ratio = args.dropout
        assert self.encoder_bert_dropout_ratio >= 0. and self.encoder_bert_dropout_ratio <= 0.5
#         self.encoder_bert_mixup = getattr(args, 'encoder_bert_mixup', False)


    def forward(self, x, encoder_padding_mask, bert_encoder_out, bert_encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        
        x = self.self_attn_layer_norm(x)

        x1, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x2, _ = self.bert_attn(query=x, key=bert_encoder_out, value=bert_encoder_out, key_padding_mask=bert_encoder_padding_mask)

        x1 = F.dropout(x1, p=self.dropout)
        x2 = F.dropout(x2, p=self.dropout)
        
        x = residual + 0.5 * x1 + 0.5 * x2
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        
        x = F.dropout(x, p=self.activation_dropout)
        x = self.fc2(x)
        
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        
        x = self.final_layer_norm(x)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, embed_tokens, no_encoder_attn=False):
        super().__init__()

        self.dropout = args.dropout
        
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.hidden_size
        self.output_embed_dim = args.hidden_size

        padding_idx = 1
        
        self.max_target_positions = args.vocab_size

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.vocab_size, embed_dim, padding_idx,
            learned=True,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for i in range(args.num_decoder_layers)
        ])

        self.adaptive_softmax = None
        args.adaptive_softmax_cutoff = None
        
        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim else None

#         if args.adaptive_softmax_cutoff is not None:
#             self.adaptive_softmax = AdaptiveSoftmax(
#                 len(dictionary),
#                 self.output_embed_dim,
#                 options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
#                 dropout=args.adaptive_softmax_dropout,
#                 adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
#                 factor=args.adaptive_softmax_factor,
#                 tie_proj=args.tie_adaptive_proj,
#             )
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(args.vocab_size, self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, bert_encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, bert_encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, bert_encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                bert_encoder_out['bert_encoder_out'],
                bert_encoder_out['bert_encoder_padding_mask'],
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, bert_gate=True):
        super().__init__()
        self.embed_dim = args.hidden_size
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.num_attention_heads,
            dropout=args.dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        
        self.dropout = args.dropout
        self.activation_fn = F.relu
        
        self.activation_dropout = args.dropout
        
        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = False
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.num_attention_heads,
                dropout=args.dropout, encoder_decoder_attention=True
            )
            bert_out_dim = args.hidden_size
            self.bert_attn = MultiheadAttention(
                self.embed_dim, args.num_attention_heads,
                kdim=bert_out_dim, vdim=bert_out_dim,
                dropout=args.dropout, encoder_decoder_attention=True
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.ffn_size)
        self.fc2 = Linear(args.ffn_size, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        
#         self.encoder_ratio = args.encoder_ratio
#         self.bert_ratio = args.bert_ratio

        self.encoder_bert_dropout = False
        self.encoder_bert_dropout_ratio = 0.25
        assert self.encoder_bert_dropout_ratio >= 0. and self.encoder_bert_dropout_ratio <= 0.5
        self.encoder_bert_mixup = False

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        bert_encoder_out=None,
        bert_encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.self_attn_layer_norm(x)
        
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}

    
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}

            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=True,
            )
            
            x2, _ = self.bert_attn(
                query=x,
                key=bert_encoder_out,
                value=bert_encoder_out,
                key_padding_mask=bert_encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=False,
            )
            
            x1 = F.dropout(x1, p=self.dropout)
            x2 = F.dropout(x2, p=self.dropout)
            
            x = residual + 0.5 * x1 + 0.5 * x2
            x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
       
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
        
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m