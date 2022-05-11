"""Custom overriding of Huggingface transformers EncoderDecoderModel.forward.
"""
import warnings

import torch
from torch import nn
import torch.distributed as dist

from transformers import EncoderDecoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation_utils import (BeamSearchEncoderDecoderOutput,
                                           BeamSearchDecoderOnlyOutput)


class MyEncoderDecoderModel(EncoderDecoderModel):

    def __init__(self,
                 config=None,
                 encoder=None,
                 decoder=None,
                 use_copy_attention=False,
                 noise_scale=0):
        super().__init__(config=config, encoder=encoder, decoder=decoder)
        self.noise_scale = noise_scale
        self.use_copy_attention = use_copy_attention
        if use_copy_attention:
            self.copy_alpha_linear = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_multitask=False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        if self.use_copy_attention:
            decoder_last_state = decoder_outputs.hidden_states[-1]  # [bs, T, E]
            copy_attn = decoder_outputs.cross_attentions[-1].mean(dim=1)
            copy_attn = copy_attn * attention_mask.unsqueeze(1)
            copy_alpha = torch.sigmoid(
                self.copy_alpha_linear(decoder_last_state))  # [bs, T, 1]
            # [bs, T, V]
            p_gen = torch.softmax(decoder_outputs.logits, dim=-1) * (1 - copy_alpha)
            p_copy = copy_alpha * copy_attn  # [bs, T, S]

            src_tokens = input_ids.unsqueeze(1).repeat(1, p_gen.shape[1], 1)
            scores = p_gen.scatter_add_(-1, src_tokens, p_copy)
            decoder_outputs.logits = torch.log(scores + 1e-12)

        if not return_dict:
            out = decoder_outputs + encoder_outputs

        out = Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        if do_multitask:
            decoder_states = decoder_outputs.hidden_states[-1]  # [bs, T, E]
            cls_token_idx = torch.where(
                decoder_input_ids == self.decoder.config.eos_token_id)
            cls_result = decoder_states[cls_token_idx]
            assert cls_result.shape == encoder_hidden_states[:, 0, :].shape
            return out, cls_result

        return out

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      past2=None,
                                      past3=None,
                                      attention_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      **kwargs):
        decoder = self.get_decoder()
        if isinstance(decoder, tuple):
            if len(decoder) == 3:
                decoder1, decoder2, decoder3 = decoder
            else:
                decoder1, decoder2 = decoder
            decoder_inputs1 = decoder1.prepare_inputs_for_generation(input_ids, past=past)
            decoder_inputs2 = decoder2.prepare_inputs_for_generation(input_ids,
                                                                     past=past2)
            if len(decoder) == 3:
                decoder_inputs3 = decoder3.prepare_inputs_for_generation(input_ids,
                                                                         past=past3)
            # checking
            assert (decoder_inputs1["input_ids"] == decoder_inputs2["input_ids"]).all()
            att_mask1 = decoder_inputs1[
                "attention_mask"] if "attention_mask" in decoder_inputs1 else None
            att_mask2 = decoder_inputs2[
                "attention_mask"] if "attention_mask" in decoder_inputs2 else None
            att_mask_same = att_mask1 == att_mask2
            if isinstance(att_mask_same, bool):
                assert att_mask_same
            else:
                assert att_mask_same.all()

            decoder_inputs = decoder_inputs1
        else:
            decoder_inputs = decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs[
            "attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "input_ids": kwargs['encoder_input_ids'] \
                if 'encoder_input_ids' in kwargs else None,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        if "encoder_outputs2" in kwargs:
            input_dict["encoder_outputs2"] = kwargs["encoder_outputs2"]
        if isinstance(decoder, tuple):
            input_dict["past_key_values2"] = decoder_inputs2["past_key_values"]
            if len(decoder) == 3:
                input_dict["past_key_values3"] = decoder_inputs3["past_key_values"]
        return input_dict

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor,
                                                       model_kwargs):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith(
                    "cross_attn") or argument == 'encoder_input_ids')
            }
            if isinstance(encoder, tuple):
                if len(encoder) == 3:
                    encoder1, encoder2, encoder3 = encoder
                else:
                    encoder1, encoder2 = encoder
                model_kwargs["encoder_outputs"] = encoder1(input_ids,
                                                           return_dict=True,
                                                           **encoder_kwargs)
                model_kwargs["encoder_outputs2"] = encoder2(input_ids,
                                                            return_dict=True,
                                                            **encoder_kwargs)
                if len(encoder) == 3:
                    model_kwargs["encoder_outputs3"] = encoder3(input_ids,
                                                                return_dict=True,
                                                                **encoder_kwargs)
            else:
                model_kwargs["encoder_outputs"] = encoder(input_ids,
                                                          return_dict=True,
                                                          **encoder_kwargs)
        return model_kwargs

    def _process_encoder_input_ids(
        self,
        beam_size,
        **kwargs,
    ):
        encoder_input_ids = kwargs['encoder_input_ids']
        expanded_return_idx = (torch.arange(encoder_input_ids.shape[0]).view(
            -1, 1).repeat(1, beam_size).view(-1).to(encoder_input_ids.device))
        encoder_input_ids = encoder_input_ids.index_select(0, expanded_return_idx)

        kwargs['encoder_input_ids'] = encoder_input_ids
        # extra for ensemble model
        if 'encoder_outputs2' in kwargs:
            kwargs['encoder_outputs2']["last_hidden_state"] = kwargs[
                'encoder_outputs2'].last_hidden_state.index_select(
                    0,
                    expanded_return_idx.to(
                        kwargs['encoder_outputs2'].last_hidden_state.device))
        if 'encoder_outputs3' in kwargs:
            kwargs['encoder_outputs3']["last_hidden_state"] = kwargs[
                'encoder_outputs3'].last_hidden_state.index_select(
                    0,
                    expanded_return_idx.to(
                        kwargs['encoder_outputs3'].last_hidden_state.device))
        return kwargs

    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processor=None,
        stopping_criteria=None,
        max_length=None,
        pad_token_id=None,
        eos_token_id=None,
        output_attentions=None,
        output_hidden_states=None,
        output_scores=None,
        return_dict_in_generate=None,
        synced_gpus=None,
        **model_kwargs,
    ):
        """
        override to get rid of log softmax
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate if return_dict_in_generate
                                   is not None else self.config.return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and
                                    output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and
                                       output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (model_kwargs["encoder_outputs"].get("hidden_states")
                                     if output_hidden_states else None)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        if 'encoder_input_ids' in model_kwargs:
            model_kwargs = self._process_encoder_input_ids(num_beams, **model_kwargs)

        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            past2 = None
            past3 = None
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    outputs, past2, past3 = outputs
                else:
                    outputs, past2 = outputs

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits,
                                                                     cur_len=cur_len)
            # NOTE: when using copy attention, no softmax should be used afterwards
            if not self.use_copy_attention:
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            else:
                next_token_scores = next_token_logits

            next_token_scores = logits_processor(input_ids, next_token_scores)
            if self.noise_scale > 0:
                beam_noise = torch.rand(batch_size, num_beams) * self.noise_scale
                beam_noise = beam_noise.view(-1).to(beam_scores.device)
                beam_scores += beam_noise
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions,)
                                           if self.config.is_encoder_decoder else
                                           (outputs.attentions,))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states,)
                                              if self.config.is_encoder_decoder else
                                              (outputs.hidden_states,))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                        2 * num_beams,
                                                        dim=1,
                                                        largest=True,
                                                        sorted=True)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :],
                 beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                past2=past2,
                past3=past3)
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
            if model_kwargs["past2"] is not None:
                model_kwargs["past2"] = self._reorder_cache(model_kwargs["past2"],
                                                            beam_idx)
            if model_kwargs["past3"] is not None:
                model_kwargs["past3"] = self._reorder_cache(model_kwargs["past3"],
                                                            beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    @staticmethod
    def _update_model_kwargs_for_generation(outputs,
                                            model_kwargs,
                                            is_encoder_decoder=False,
                                            past2=None,
                                            past3=None):
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None
        model_kwargs["past2"] = past2
        model_kwargs["past3"] = past3

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                                                           dim=-1)

        return model_kwargs
