import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import (BertConfig, GPT2Config, EncoderDecoderConfig, ElectraModel)
from transformers_encoder_decoder import MyEncoderDecoderModel


def get_model(args, tokenizer, annotation_size=8):
    if args.tokenizer == 'char':
        vocab_size = args.vocab_size
    else:
        vocab_size = len(tokenizer)
    if args.model == 'transformer':
        return TransformerModel(vocab_size,
                                args.hidden_size,
                                args.num_attention_heads,
                                args.num_encoder_layers,
                                args.num_decoder_layers,
                                args.intermediate_size,
                                anntation_size=annotation_size,
                                dropout=args.dropout,
                                share_embedding=args.share_embedding,
                                use_copy_attention=args.use_copy_attention,
                                sos_idx=tokenizer.sos_idx,
                                eos_idx=tokenizer.eos_idx)
    if args.model in ('bert_gpt', 'ensemble'):  # ensemble is for bert gpt models
        return BertGPT2Model(vocab_size,
                             args.hidden_size,
                             args.num_attention_heads,
                             args.num_encoder_layers,
                             args.num_decoder_layers,
                             args.intermediate_size,
                             anntation_size=annotation_size,
                             dropout=args.dropout,
                             share_embedding=args.share_embedding,
                             use_copy_attention=args.use_copy_attention,
                             sos_idx=tokenizer.sos_idx,
                             eos_idx=tokenizer.eos_idx,
                             noise_scale=args.noise_scale)
    if args.model == 'electra_gpt':
        return ElectraGPT2Model(vocab_size,
                                args.num_attention_heads,
                                args.num_decoder_layers,
                                anntation_size=annotation_size,
                                dropout=args.dropout,
                                share_embedding=args.share_embedding,
                                use_copy_attention=args.use_copy_attention,
                                sos_idx=tokenizer.sos_idx,
                                eos_idx=tokenizer.eos_idx)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = PositionalEncoding(embedding_size)

        self.embedding_size = embedding_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(
            self.embedding_size) + self.positional_encoder(x)


class TransformerModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_attention_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 intermediate_size,
                 anntation_size=8,
                 dropout=0.1,
                 share_embedding=False,
                 use_copy_attention=False,
                 sos_idx=2,
                 eos_idx=3):
        super().__init__()
        del sos_idx, eos_idx  # unused
        self.share_embedding = share_embedding
        self.use_copy_attention = use_copy_attention

        self.token_embeddings = TransformerEmbedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout)

        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=intermediate_size,
            dropout=dropout,
        )
        if not share_embedding:
            self.decoder_embeddings = nn.Linear(embedding_size, vocab_size)

        self.cls_head = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                      nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(embedding_size, anntation_size))

        if use_copy_attention:
            self.copy_attention = nn.MultiheadAttention(embedding_size,
                                                        num_attention_heads,
                                                        dropout=dropout)
            self.copy_alpha_linear = nn.Linear(embedding_size, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0,
                                        float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, initrange=0.1):
        self.token_embeddings.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.share_embedding:
            self.decoder_embeddings.bias.data.zero_()
            self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src=None,
                tgt=None,
                memory=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                do_multitask=False):
        """
        src: [bs, S]
        tgt: [bs, T]
        memory: [S, bs, E]
        src_key_padding_mask: [bs, S]
        tgt_key_padding_mask: [bs, T]
        memory_key_padding_mask: [bs, S]
        cls_tokens: [bs, 1]
        """
        if src is not None and memory is None:
            src = src.t()  # [S, bs]
            src_embeddings = self.dropout(self.token_embeddings(src))
            memory = self.transformer.encoder(  # [S, bs, E]
                src_embeddings, src_key_padding_mask=src_key_padding_mask)
            if tgt is None:  # encode
                return memory
        else:
            assert memory is not None, \
                'when src is not provided, memory must be provided.'

        assert tgt is not None, \
            'src and tgt cannot both be None.'

        tgt = tgt.t()  # [T, bs]
        tgt_embeddings = self.dropout(self.token_embeddings(tgt))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(
            tgt.device)  # [T, T]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        output = self.transformer.decoder(
            tgt_embeddings,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)  # [T, bs, E]
        if self.use_copy_attention:
            x_copy, copy_attn = self.copy_attention(
                output, memory, memory, key_padding_mask=memory_key_padding_mask)
            x_copy = x_copy.permute(1, 0, 2)  # [bs, T, E]
            copy_alpha = torch.sigmoid(self.copy_alpha_linear(x_copy))  # [bs, T, 1]
            # [bs, T, S]
            copy_attn = torch.clamp(copy_attn, 0, 1)  # fix neg loss bug

        if not self.share_embedding:
            output = self.decoder_embeddings(output)  # [T, bs, V]
        else:
            output = F.linear(output, self.token_embeddings.embedding.weight)

        if self.use_copy_attention:
            # [bs, T, S]
            src_tokens = src.t().unsqueeze(1).repeat(1, output.shape[0], 1)

            scores = F.softmax(output.permute(1, 0, 2), dim=-1)  # [bs, T, V]

            scores = (1 - copy_alpha) * scores
            copy_scores = copy_alpha * copy_attn  # [bs, T, S]
            scores.scatter_add_(-1, src_tokens, copy_scores)
            scores = torch.log(scores + 1e-12)
            output = scores.permute(1, 0, 2)  # [T, bs, V]
        else:
            output = F.log_softmax(output, dim=-1)

        # multitask:
        if do_multitask:
            cls_emb = memory[0, :, :]  # [bs, E]
            cls_result = self.cls_head(cls_emb)
            return output, cls_result

        return output

    def generate(self, src_padded, src_padding_mask, sos_idx, eos_idx):
        # [T, bs, V]
        memory = self(src=src_padded, src_key_padding_mask=src_padding_mask)

        tgt_token_ids = [[sos_idx] for _ in range(src_padded.shape[0])]
        end = [False for _ in range(src_padded.shape[0])]
        src_max_seq_length = src_padded.shape[1]
        for l in range(src_max_seq_length + 20):
            tgt = torch.LongTensor(tgt_token_ids).contiguous().to(memory.device)
            output = self(
                src=src_padded.t(),  # transpose to match the shape [S, bs]
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_padding_mask)  # [T, bs, V]
            top1 = output[-1].argmax(-1).tolist()  # [bs]
            for i, tok in enumerate(top1):
                src_seq_length = (~src_padding_mask[i]).sum()
                if tok == eos_idx or l >= src_seq_length + 20:
                    end[i] = True
                tgt_token_ids[i].append(tok if not end[i] else eos_idx)
            if all(end):
                break

        return tgt_token_ids


class HFEncoderDecoderBase(nn.Module):

    def forward(self,
                src,
                src_key_padding_mask,
                tgt,
                tgt_key_padding_mask,
                do_multitask=False):
        source_mask = (~src_key_padding_mask).long()
        target_mask = (~tgt_key_padding_mask).long()
        outputs = self.model(input_ids=src,
                             attention_mask=source_mask,
                             decoder_input_ids=tgt,
                             decoder_attention_mask=target_mask,
                             do_multitask=do_multitask,
                             output_attentions=True,
                             output_hidden_states=True)
        if do_multitask:
            outputs, pooled_out = outputs
        logits = outputs.logits  # [bs, T, V]
        logits = logits.permute(1, 0, 2)  # [T, bs, V]
        if not self.model.use_copy_attention:
            logits = F.log_softmax(logits, dim=-1)

        if do_multitask:
            cls_result = self.cls_head(pooled_out)
            return logits, cls_result
        return logits

    @torch.no_grad()
    def generate(self,
                 src_padded,
                 src_padding_mask,
                 sos_idx,
                 eos_idx,
                 pad_idx,
                 for_pred=True):
        source_mask = (~src_padding_mask).long()
        src_seq_len = src_padded.shape[1]
        generated = self.model.generate(
            input_ids=src_padded,
            attention_mask=source_mask,
            max_length=src_seq_len + 20,
            bos_token_id=sos_idx,
            eos_token_id=eos_idx,
            pad_token_id=pad_idx,  # for open end generation
            num_beams=4,  # beam search
            # no_repeat_ngram_size=2,
            early_stopping=True,
            output_attentions=True,
            output_hidden_states=True,
            encoder_input_ids=src_padded)  # [bs, T]
        if for_pred:
            generated = generated.tolist()

        return generated


class EnsembleModel(MyEncoderDecoderModel):

    def __init__(self, model1, model2):
        super().__init__(config=model1.model.config,
                         encoder=model1.model.encoder,
                         decoder=model1.model.decoder,
                         use_copy_attention=model1.model.use_copy_attention,
                         noise_scale=model1.model.noise_scale)
        self.model1 = model1.model
        self.model2 = model2.model
        self.config = self.model1.config
        self.encoder = self.model1.encoder
        self.decoder = self.model1.decoder

    def get_encoder(self):
        return self.model1.get_encoder(), self.model2.get_encoder()

    def get_decoder(self):
        return self.model1.get_decoder(), self.model2.get_decoder()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_outputs=None,
                encoder_outputs2=None,
                past_key_values=None,
                past_key_values2=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        model1_out = self.model1(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_attention_mask,
                                 encoder_outputs=encoder_outputs,
                                 past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds,
                                 decoder_inputs_embeds=decoder_inputs_embeds,
                                 labels=labels,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 do_multitask=False,
                                 **kwargs)

        model2_out = self.model2(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_attention_mask,
                                 encoder_outputs=encoder_outputs2,
                                 past_key_values=past_key_values2,
                                 inputs_embeds=inputs_embeds,
                                 decoder_inputs_embeds=decoder_inputs_embeds,
                                 labels=labels,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 do_multitask=False,
                                 **kwargs)

        model1_out.logits = (model1_out.logits + model2_out.logits) / 2

        return model1_out, model2_out.past_key_values

    @torch.no_grad()
    def generate(self,
                 src_padded,
                 src_padding_mask,
                 sos_idx,
                 eos_idx,
                 pad_idx,
                 for_pred=True):
        source_mask = (~src_padding_mask).long()
        src_seq_len = src_padded.shape[1]
        generated = super().generate(
            input_ids=src_padded,
            attention_mask=source_mask,
            max_length=src_seq_len + 20,
            bos_token_id=sos_idx,
            eos_token_id=eos_idx,
            pad_token_id=pad_idx,  # for open end generation
            num_beams=4,  # beam search
            # no_repeat_ngram_size=2,
            early_stopping=True,
            output_attentions=True,
            output_hidden_states=True,
            encoder_input_ids=src_padded,
        )  # [bs, T]
        if for_pred:
            generated = generated.tolist()

        return generated


class Ensemble3Model(MyEncoderDecoderModel):

    def __init__(self, model1, model2, model3):
        super().__init__(config=model1.model.config,
                         encoder=model1.model.encoder,
                         decoder=model1.model.decoder,
                         use_copy_attention=model1.model.use_copy_attention,
                         noise_scale=model1.model.noise_scale)
        self.model1 = model1.model
        self.model2 = model2.model
        self.model3 = model3.model
        self.config = self.model1.config
        self.encoder = self.model1.encoder
        self.decoder = self.model1.decoder

    def get_encoder(self):
        return self.model1.get_encoder(), self.model2.get_encoder(
        ), self.model3.get_encoder()

    def get_decoder(self):
        return self.model1.get_decoder(), self.model2.get_decoder(
        ), self.model3.get_decoder()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_outputs=None,
                encoder_outputs2=None,
                encoder_outputs3=None,
                past_key_values=None,
                past_key_values2=None,
                past_key_values3=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        model1_out = self.model1(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_attention_mask,
                                 encoder_outputs=encoder_outputs,
                                 past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds,
                                 decoder_inputs_embeds=decoder_inputs_embeds,
                                 labels=labels,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 do_multitask=False,
                                 **kwargs)

        model2_out = self.model2(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_attention_mask,
                                 encoder_outputs=encoder_outputs2,
                                 past_key_values=past_key_values2,
                                 inputs_embeds=inputs_embeds,
                                 decoder_inputs_embeds=decoder_inputs_embeds,
                                 labels=labels,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 do_multitask=False,
                                 **kwargs)
        model3_out = self.model3(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_attention_mask,
                                 encoder_outputs=encoder_outputs3,
                                 past_key_values=past_key_values3,
                                 inputs_embeds=inputs_embeds,
                                 decoder_inputs_embeds=decoder_inputs_embeds,
                                 labels=labels,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,
                                 return_dict=return_dict,
                                 do_multitask=False,
                                 **kwargs)

        model1_out.logits = (model1_out.logits + model2_out.logits +
                             model3_out.logits) / 3

        return model1_out, model2_out.past_key_values, model3_out.past_key_values

    @torch.no_grad()
    def generate(self,
                 src_padded,
                 src_padding_mask,
                 sos_idx,
                 eos_idx,
                 pad_idx,
                 for_pred=True):
        source_mask = (~src_padding_mask).long()
        src_seq_len = src_padded.shape[1]
        generated = super().generate(
            input_ids=src_padded,
            attention_mask=source_mask,
            max_length=src_seq_len + 20,
            bos_token_id=sos_idx,
            eos_token_id=eos_idx,
            pad_token_id=pad_idx,  # for open end generation
            num_beams=4,  # beam search
            # no_repeat_ngram_size=2,
            early_stopping=True,
            output_attentions=True,
            output_hidden_states=True,
            encoder_input_ids=src_padded,
        )  # [bs, T]
        if for_pred:
            generated = generated.tolist()

        return generated


class BertGPT2Model(HFEncoderDecoderBase):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_attention_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 intermediate_size,
                 anntation_size=8,
                 dropout=0.1,
                 share_embedding=False,
                 use_copy_attention=False,
                 sos_idx=2,
                 eos_idx=3,
                 noise_scale=6):
        super().__init__()
        # TODO: implement multitask, copy attention

        encoder_config = BertConfig(vocab_size=vocab_size,
                                    hidden_size=embedding_size,
                                    num_hidden_layers=num_encoder_layers,
                                    num_attention_heads=num_attention_heads,
                                    intermediate_size=intermediate_size,
                                    hidden_dropout_prob=dropout,
                                    attention_probs_dropout_prob=dropout)
        decoder_config = GPT2Config(vocab_size=vocab_size,
                                    n_embd=embedding_size,
                                    n_layer=num_decoder_layers,
                                    n_head=num_attention_heads,
                                    resid_pdrop=dropout,
                                    embd_pdrop=dropout,
                                    attn_pdrop=dropout,
                                    bos_token_id=sos_idx,
                                    eos_token_id=eos_idx)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config)

        self.model = MyEncoderDecoderModel(config,
                                           use_copy_attention=use_copy_attention,
                                           noise_scale=noise_scale)

        # use the same embedding in encoder and decoder
        self.model.decoder.transformer.wte.weight = self.model.encoder.embeddings.word_embeddings.weight

        if share_embedding:  # decoder output use tied embedding
            self.model.decoder.lm_head.weight = self.model.decoder.transformer.wte.weight

        self.cls_head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(embedding_size, anntation_size))


class ElectraGPT2Model(HFEncoderDecoderBase):
    r"""
    ElectraModel retrieved from the implementation provided by monologg.

    @misc{park2020koelectra,
          author = {Park, Jangwon},
          title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
          year = {2020},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/monologg/KoELECTRA}}
          }
    """

    def __init__(self,
                 vocab_size,
                 num_attention_heads,
                 num_decoder_layers,
                 anntation_size=8,
                 dropout=0.1,
                 share_embedding=False,
                 use_copy_attention=False,
                 sos_idx=2,
                 eos_idx=3):
        super().__init__()
        # NOTE: This model is not compatible with the new version: since Electra
        # do not have pooled output. Modify transformers_encoder_decoder.py

        encoder = ElectraModel.from_pretrained(
            'monologg/koelectra-base-v3-discriminator')  # big
        encoder.resize_token_embeddings(vocab_size)
        decoder_config = GPT2Config(vocab_size=vocab_size,
                                    n_embd=encoder.config.embedding_size,
                                    n_layer=num_decoder_layers,
                                    n_head=num_attention_heads,
                                    resid_pdrop=dropout,
                                    embd_pdrop=dropout,
                                    attn_pdrop=dropout,
                                    bos_token_id=sos_idx,
                                    eos_token_id=eos_idx)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder_config)

        self.model = MyEncoderDecoderModel(config,
                                           encoder=encoder,
                                           use_copy_attention=use_copy_attention)

        # use the same embedding in encoder and decoder
        self.model.decoder.transformer.wte = self.model.encoder.embeddings.word_embeddings

        if share_embedding:  # decoder output use tied embedding
            self.model.decoder.lm_head.weight = self.model.decoder.transformer.wte.weight

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder.config.embedding_size, anntation_size))
