import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.generation.logits_process import LogitsProcessorList, LogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from pdb import set_trace

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]


@dataclass
class CausalLMOutputWithPast_val(CausalLMOutputWithPast):
    
    loss_txt: Optional[torch.FloatTensor] = None
    loss_txt_1: Optional[torch.FloatTensor] = None
    loss_num: Optional[torch.FloatTensor] = None


@dataclass
class GenerateDecoderOnlyOutput_val(GenerateDecoderOnlyOutput):
    is_num: List[bool] = None


class TemperatureLogitsWarper_val(LogitsWarper):
    def __init__(self, temperature: float, temperature_val: float, val_start_id: int, val_end_id: int):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature
        self.temperature_val = temperature_val
        self.val_start_id = val_start_id
        self.val_end_id = val_end_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        if self.temperature_val != self.temperature:
            scores_val = scores[:, self.val_start_id:self.val_end_id] / self.temperature_val
            scores_val = scores_val + scores_processed[:, self.val_start_id:self.val_end_id].logsumexp(dim=-1, keepdim=True) \
                        - scores_val.logsumexp(dim=-1, keepdim=True)
            scores_processed[:, self.val_start_id:self.val_end_id] = scores_val
        return scores_processed


class LlamaForCausalLM_val(LlamaForCausalLM):

    def mask_prosody(self, input_ids):
        #assert input_ids.size(0) == 1
        # 1) Create start_mask and end_mask (shape (1, n))
        start_mask = (input_ids == self.config.sep1_token_id).int()  # +1 at each integer1
        end_mask   = (input_ids == self.config.sep2_token_id).int()  # +1 at each integer2
        # 2) Compute cumulative sums along dimension=1
        cumulative_starts = start_mask.cumsum(dim=1)
        cumulative_ends   = end_mask.cumsum(dim=1)
        # 3) Shift cumulative_ends by one position along dimension=1.
        #    This ensures that 'integer2' is still considered "inside" at its own index.
        zeros_to_cat = torch.zeros((input_ids.size(0), 1), 
                                   dtype=cumulative_ends.dtype, device=input_ids.device) 
        cumulative_ends_excl = torch.cat([zeros_to_cat, cumulative_ends[:, :-1]], dim=1)
        # 4) Determine which positions are inside any (integer1 ... integer2) pair
        inside = (cumulative_starts - cumulative_ends_excl) > 0  # Boolean mask (shape (1, n))
        # 5) Optionally flip the mask if you want the "outside" to be True
        mask_flipped = ~inside
        return mask_flipped

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #input_ids = input_ids.new(1, 4600).random_(0, self.config.spk_glb_start_id+1024)
        #attention_mask = attention_mask.new(1, 4600).fill_(1)
        #labels = input_ids.clone()

        if labels is not None:
            mask_txt = self.mask_prosody(input_ids)
            mask_num = (input_ids >= self.config.num_start_id)
            assert not mask_num[:, 0].any()
            num_sils = (input_ids == self.config.sil_token_id).sum()
            mask_valid = (input_ids != self.config.invalid_num_id)
            num_sents = (input_ids == self.config.sep1_token_id).sum()
            assert (mask_num.sum() - num_sils - num_sents*0 - 1) % 5 == 0
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            set_trace()
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss, loss_txt, loss_txt_1, loss_num = None, None, None, None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            mask_inp = (shift_labels != -100)
            mask_num = mask_num[..., 1:].reshape(-1)
            mask_valid = mask_valid[..., 1:].reshape(-1)
            mask_txt = mask_txt[..., 1:].reshape(-1)
            loss_txt = loss[(~mask_num) & mask_inp].mean()
            loss_txt_1 = loss[(~mask_num) & mask_inp & mask_txt].mean()
            loss_num = loss[mask_num & mask_inp & mask_valid].mean()
            if torch.isnan(loss_num):
                set_trace()
            loss = loss_txt + loss_num * self.config.lambda_num

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast_val(
            loss=loss,
            loss_txt=loss_txt,
            loss_txt_1=loss_txt_1,
            loss_num=loss_num,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        guided = False
        output_ids = generation_config.output_ids
        if output_ids is not None:
            guided = True
        
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if generation_config.temperature_val is not None:
            logits_processor[1] = TemperatureLogitsWarper_val(generation_config.temperature,
                                                              generation_config.temperature_val,
                                                              generation_config.num_start_id,
                                                              generation_config.num_end_id)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        is_num = [] if (return_dict_in_generate and output_logits) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        assert batch_size == 1
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        p = 0
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()

            # pre-process distribution
            if guided and output_ids[p] < generation_config.num_start_id:
                next_token_logits = torch.zeros_like(next_token_logits)
                next_token_logits[:, output_ids[p]] = 1000
                next_token_scores = next_token_logits
            else:
                next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if guided and output_ids[p] < generation_config.num_start_id:
                assert next_tokens.item() == output_ids[p]

            if return_dict_in_generate:
                if output_logits:
                    if next_tokens.item() >= generation_config.num_start_id:
                        is_num.append(True)
                    else:
                        is_num.append(False)
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            p += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if return_dict_in_generate:
            if output_logits:
                assert len(is_num)==len(raw_logits)
            if self.config.is_encoder_decoder:
                raise NotImplementedError("self.config.is_encoder_decoder is True")
            else:
                return GenerateDecoderOnlyOutput_val(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    is_num=is_num,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


# Replace the expanded embedding layer with a custom one
class FreezableEmbedding(nn.Module):
    def __init__(self, original_embeddings, num_new_tokens):
        super().__init__()
        self.original_vocab_size = original_embeddings.shape[0]
        self.embed_dim = original_embeddings.shape[1]
        self.dtype = original_embeddings.dtype
        
        # Original embeddings (frozen)
        self.original_embed = nn.Embedding.from_pretrained(original_embeddings, freeze=True)
        
        # New embeddings (trainable)
        self.new_embed = nn.Embedding(num_new_tokens, self.embed_dim)
        self.new_embed = self.new_embed.to(dtype=self.dtype)
        nn.init.normal_(self.new_embed.weight, std=0.02)

    def forward(self, input_ids):
        # Separate original and new token IDs
        original_mask = (input_ids < self.original_vocab_size)
        new_mask = ~original_mask
        
        # Combine embeddings
        embeddings = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.embed_dim,
            dtype=self.dtype,
            device=input_ids.device
        )
        if original_mask.any():
            embeddings[original_mask] = self.original_embed(input_ids[original_mask])
        if new_mask.any():
            new_ids = input_ids[new_mask] - self.original_vocab_size
            embeddings[new_mask] = self.new_embed(new_ids)
        return embeddings


class FreezableLMHead(nn.Module):
    def __init__(self, original_weight, num_new_tokens):
        super().__init__()
        self.original_vocab_size = original_weight.size(0)
        self.hidden_size = original_weight.size(1)
        self.dtype = original_weight.dtype
        
        # Original token weights (frozen)
        self.original_head = nn.Linear(self.hidden_size, self.original_vocab_size, bias=False)
        with torch.no_grad():
            self.original_head.weight.copy_(original_weight)
        self.original_head = self.original_head.to(dtype=self.dtype)
        self.original_head.weight.requires_grad = False
        
        # New token weights (trainable, using nn.Linear)
        self.new_head = nn.Linear(self.hidden_size, num_new_tokens, bias=False)
        self.new_head = self.new_head.to(dtype=self.dtype)
        nn.init.normal_(self.new_head.weight, std=0.02)
    
    def forward(self, hidden_states):
        # Compute logits for original and new tokens separately
        original_logits = self.original_head(hidden_states)
        new_logits = self.new_head(hidden_states)
        # Concatenate results
        return torch.cat([original_logits, new_logits], dim=-1)