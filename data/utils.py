import re
import sys
from functools import lru_cache
from phonemizer.backend import EspeakBackend
from .w2w_mapper import convert_with_word_level


_suprasegmentals = 'ˈˌːˑ'
_punctuation = '.!;:,?—'


class Phonemizer():
    def __init__(self):
        self.backend = EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            words_mismatch="ignore",
        )
        
    @lru_cache(maxsize=1000)
    def text_to_phonemes(self, text):
        outputs = self.backend.phonemize([text])
        output = outputs[0] if len(outputs) else ""

        # Correct leading/trailing spaces
        if text[:1] == " " and output[:1] != " ":
            output = " " + output
        if text[:1] != " " and output[:1] == " ":
            output = output[1:]
        if text[-1:] == " " and output[-1:] != " ":
            output = output + " "
        if text[-1:] != " " and output[-1:] == " ":
            output = output[:-1]

        # Phonemizer may introduce spaces before punctuation, so we remove them.
        j = 0
        while j < len(output) - 1:
            if output[j] == " " and output[j + 1] in _punctuation:
                output = output[:j] + output[j + 1:]
            j += 1

        return output


@lru_cache()
def _get_backend(language: str):
    """
    Other backend and parameter combinations have not been tested, but I assume most would work.
    """
    return EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        words_mismatch="ignore",
    )


@lru_cache()
def post_process(ps: str):
    phon_groups = ps.split(' ')
    phon_groups_new = []
    for p_g in phon_groups:
        if bool(re.search(r'(?<!^)[.,!?;:\'"—](?!$|[.,!?;:\'"—])', p_g)):
            p_g = re.sub(r'(?<!\s)(?<!^)([.,!?;:\'"—])(?!\s)(?!$)', r'\1 ', p_g)
            p_g = p_g.split(' ')
            phon_groups_new.extend(p_g)
        else:
            phon_groups_new.append(p_g)
    return " ".join(phon_groups_new)


@lru_cache(maxsize=1000)
def _text_to_phonemes(text: str, language="en-us"):
    """
    This function wraps phonemize() and ensures that punctuation and spaces are as consistent as possible through
    conversion.
    """
    # Phonemize
    backend = _get_backend(language)
    outputs = backend.phonemize([text])
    output = outputs[0] if len(outputs) else ""

    # Correct leading/trailing spaces
    if text[:1] == " " and output[:1] != " ":
        output = " " + output
    if text[:1] != " " and output[:1] == " ":
        output = output[1:]
    if text[-1:] == " " and output[-1:] != " ":
        output = output + " "
    if text[-1:] != " " and output[-1:] == " ":
        output = output[:-1]

    # Phonemizer may introduce spaces before punctuation, so we remove them.
    j = 0
    while j < len(output) - 1:
        if output[j] == " " and output[j + 1] in _punctuation:
            output = output[:j] + output[j + 1:]
        j += 1

    output = post_process(output)
    return output


def word_mappings(file_stem, text):
    # Define the equality and the conversion functions for the word-level mapper
    remove_supresegmentals = lambda s: "".join(c for c in s if not c in _suprasegmentals)
    eq_fn = lambda x, y: remove_supresegmentals(x) == remove_supresegmentals(y)
    conv_fn = lambda t: _text_to_phonemes(t)

    return convert_with_word_level(file_stem, text, conv_fn, eq_fn)


def find_consecutive_indices(arr):
    result = []
    start = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            result.append((arr[start], start, i))  # Store (value, start_index, end_index)
            start = i
    # Add the last sequence
    result.append((arr[start], start, len(arr)))
    return result

