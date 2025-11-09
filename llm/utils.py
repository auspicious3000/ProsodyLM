import re
import torch
import numpy as np


def replace_num_tokens(text):
    text = re.sub(r"<\|num_tk_\d+\|>", "@", text)
    return text


class TemperatureLogitsProcessor:
    def __init__(self, temperature: float, temperature_val: float, val_start_id: int, val_end_id: int):
        self.temperature = temperature
        self.temperature_val = temperature_val
        self.val_start_id = val_start_id
        self.val_end_id = val_end_id

    def __call__(self, token_ids: tuple, logits: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = logits / self.temperature
        if self.temperature_val != self.temperature:
            scores_val = logits[self.val_start_id:self.val_end_id] / self.temperature_val
            scores_val = scores_val + scores_processed[self.val_start_id:self.val_end_id].logsumexp(dim=-1, keepdim=True) \
                        - scores_val.logsumexp(dim=-1, keepdim=True)
            scores_processed[self.val_start_id:self.val_end_id] = scores_val
        return scores_processed


class RangeConstrainedLogitsProcessor:
    def __init__(self, allowed_token_ids, sil_token_id):
        self.allowed_token_ids = set(allowed_token_ids)
        self.sil_token_id = sil_token_id
        self.counter = 0

    def __call__(self, input_ids, logits):
        if len(input_ids) < 2:
            return logits

        prev_token_id = input_ids[-2]
        current_token_id = input_ids[-1]
        
        if self.counter == 0 and current_token_id in self.allowed_token_ids and prev_token_id != self.sil_token_id:
            self.counter = 1
        elif self.counter > 0:
            self.counter += 1

        if 0 < self.counter < 5:
            mask = torch.full_like(logits, float('-inf'))
            mask[list(self.allowed_token_ids)] = 0
            logits += mask
        elif self.counter == 5:
            mask = torch.full_like(logits, float('-inf'))
            disallowed_ids = set(range(logits.size(-1))) - self.allowed_token_ids
            mask[list(disallowed_ids)] = 0
            logits += mask
            self.counter = 0

        return logits


def parse_prs_token_sequence(text):
    """
    Parses a specially-formatted text containing <SIL><|num_tk_X|>Word<|num_tk_...|>×5 structures,
    and requires the text to end with <SIL><|num_tk_X|>[SEP_2].

    Returns:
        A nested list in the format: [[sil_id], [5 token IDs], ..., [final_sil_id]]

    Raises:
        ValueError if:
            - The sentence start format is invalid
            - Any word is not followed by exactly 5 tokens
            - The text does not end with [SEP_2]
    """
    # 1. Check the sentence starts correctly
    if not re.match(r"^<SIL>", text):
        raise ValueError("Invalid sentence start format.")

    # 2. Check it ends with [SEP_2]
    if not text.endswith("[SEP_2]"):
        raise ValueError("Text must end with [SEP_2].")

    result = []

    # 3. Match all normal <SIL><|num_tk_X|>word<|num_tk_Y|>×5 patterns
    full_pattern = re.compile(r"<SIL><\|num_tk_(\d+)\|>([\w' ]+?)((?:<\|num_tk_(\d+)\|>){5})")
    for match in full_pattern.finditer(text):
        sil_number = int(match.group(1))
        following_numbers = re.findall(r"<\|num_tk_(\d+)\|>", match.group(3))
        if len(following_numbers) != 5:
            raise ValueError(f"Word '{match.group(2)}' does not have exactly 5 following <|num_tk_x|> tokens.")
        result.append([sil_number])
        result.append(list(map(int, following_numbers)))

    # 4. Match the final <SIL><|num_tk_X|>[SEP_2]
    trailing_match = re.search(r"<SIL><\|num_tk_(\d+)\|>\[SEP_2\]$", text)
    if not trailing_match:
        raise ValueError("Final [SEP_2] must be preceded by <SIL><|num_tk_X|>.")
    result.append([int(trailing_match.group(1))])

    return result


class TemplateLogitsProcessor:
    
    def __init__(self,
                 template,
                 pattern,
                 free_start_1, free_end_1):
        
        self.template = template
        self.pattern     = pattern
        self.free_start_1  = free_start_1
        self.free_end_1    = free_end_1

    def __call__(self,
                 past_ids,
                 logits):
        pos = len(past_ids)

        if pos >= len(self.pattern):
            logits.fill_(float('-inf'))
            return logits

        fixed_id = self.pattern[pos]

        if fixed_id is None:
            logits[:self.free_start_1]  = float('-inf')
            logits[self.free_end_1:]    = float('-inf')
            return logits

        logits.fill_(float('-inf'))
        logits[fixed_id] = 0.0
        return logits