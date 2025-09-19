import logging


def convert_with_word_level(file_stem, text: str, conv_fn, eq_fn=None):
    """
    Given text and a text-conversion function (e.g. phonemize), computes the output and maps it to the input at the
    word level.

    :param text: a single sentence to convert
    :param conv_fn: a text to text conversion function. It takes a text as input and returns the converted text as
    output. It must hold that:
        - No words are created from nothing in conversion
        - The order of converted words corresponds to the order of the words in the text
    :param eq_fn: an equality function for comparing words in the converted domain. Defaults to string compare.
    :return:
        - text_groups: the list of text groups. It holds that " ".join(text_groups) == text
        - conv: the result of the the conversion function on the entire text, i.e. conv_fn(text).
        - conv_groups: the list of groups for the converted text. It holds that text_groups[i] maps to
        conv_groups[i], and that " ".join(g for g in conv_groups if g is not None) == conv. A group with value None
        implies that the corresponding text group maps to nothing in the converted output.
    """
    eq_fn = eq_fn or (lambda x, y: x == y)

    # Get the converted output of the complete text and split both on spaces
    conv = conv_fn(text)
    text_words, conv_words = text.split(" "), conv.split(" ")

    # Find the mapping
    mapping = [(0, 0)]
    while not (mapping[-1][0] == len(text_words) and mapping[-1][1] == len(conv_words)):
        
        text_range, conv_range = _wl_sweep_search(mapping, text_words, conv_words, conv_fn, eq_fn,
                                                  max_prev_groups=30, max_forward_range=50)
        assert text_range, f"Internal error for text \"{file_stem}   {text}\""

        while True:
            if text_range == 1 or conv_range <= 1:
                # 1-x, x-1 or x-0 groups: optimal group, move on
                mapping.append((mapping[-1][0] + text_range, mapping[-1][1] + conv_range))
                break
            elif text_range == 2 and conv_range == 2:
                # 2-2 groups: a trivial case of pigeonhole principle: such a group is always separable.
                mapping += [
                    (mapping[-1][0] + 1, mapping[-1][1] + 1),
                    (mapping[-1][0] + 2, mapping[-1][1] + 2),
                ]
                break
            else:
                # The group is suboptimal: find a break point inside the group with an exhaustive search
                mapping, text_range, conv_range = _wl_backtracking_search(
                    mapping, text_words, text_range, conv_words, conv_range, conv_fn, eq_fn
                )
                if not text_range:
                    logging.warning("Word-level mapper: got suboptimal solution")
                    break

    # Get the text and conv groups based on the mapping
    text_groups, conv_groups = [], []
    for (text_start, conv_start), (text_end, conv_end) in zip(mapping, mapping[1:]):
        text_groups.append(" ".join(text_words[text_start:text_end]))
        conv_groups.append(" ".join(conv_words[conv_start:conv_end]) if conv_start != conv_end else None)
    assert " ".join(text_groups) == text and " ".join(g for g in conv_groups if g is not None) == conv, \
        f"Internal error for text \"{text}\""

    return text_groups, conv, conv_groups


def _sweep_search_params_generator(mapping, n_text_words, max_prev_groups, max_forward_range):
    """
    Generates forward and backward parameter values for the sweep search.
    """
    max_prev_groups = min(max_prev_groups, len(mapping) - 1)
    max_forward_range = min(max_forward_range, n_text_words - mapping[-1][0])
    for i in range(1, n_text_words + 1):
        forward = min(i, max_forward_range)
        backward = min(i // 2, max_prev_groups)
        yield backward, forward
        if backward == max_prev_groups and forward == max_forward_range:
            break


def _wl_sweep_search(mapping, text_words, conv_words, conv_fn, eq_fn, max_prev_groups=4, max_forward_range=8):
    """
    In a sweep search, we seek for the next group in the sequence. We are given the lists of words in the text and
    in the converted output, as well as the mapping computed so far. Starting from the last position given in the
    mapping, we take words in the text and see if their conversion matches the provided converted words. If that is
    the case, a group is found. Otherwise, words coming before or after the group must influence the conversion,
    and thus we expand our range to include them. With bounds high enough for the search parameters, this function is
    guaranteed to return a correct group.

    :param max_prev_groups: the maximum number of previous groups defined in the mapping to include in our search.
    :param max_forward_range: the maximum number of upcoming words to include in our search
    :return: the sizes of the group found
        - text_range: the number of text words in the group, starting from mapping[-1][0]. It holds that
        1 <= text_range <= max_forward_range
        - conv_range: the number of converted words in the group, starting from mapping[-1][1]. It holds that
        0 <= conv_range
    """
    # This function will generate the values <backward> and <forward>. The first iteration always returns (0, 1).
    params_generator = _sweep_search_params_generator(mapping, len(text_words), max_prev_groups, max_forward_range)

    # We perform a search for each pair of search parameters, stopping at the first valid solution
    for backward, forward in params_generator:
        # We get the starting position in both the text words and the converted words. <backward> indicates how many
        # of the previous groups we include.
        start_pos = mapping[-backward - 1]

        # We take all the text words from the groups included, plus the <forward> upcoming words. We then take the
        # conversion for theses words alone.
        text_part = " ".join(text_words[start_pos[0]:mapping[-1][0] + forward])
        conv_guess = conv_fn(text_part)

        # We compare this conversion with the actual words taken from the full conversion
        conv_range = min(conv_guess.count(" ") + 1, len(conv_words) - start_pos[1])
        conv_part = " ".join(conv_words[start_pos[1]:start_pos[1] + conv_range])
        if eq_fn(conv_part, conv_guess):
            return forward, conv_range - mapping[-1][1] + start_pos[1]

    # In case the search parameters are not large enough, the function may fail to find a group.
    return None, None


def _backtracking_group_generator(text_range, conv_range):
    """
    Generates group guesses for the backtracking search
    """
    for total_group_size in range(2, text_range + conv_range):
        for i in range(1, total_group_size):
            group = (i, total_group_size - i)
            if group[0] < text_range and group[1] < conv_range:
                yield group


def _wl_backtracking_search(mapping, text_words, text_range, conv_words, conv_range, conv_fn, eq_fn):
    """
    When a group is suboptimal (x-y group with x >= 2 and y >= 2), we makes guesses as to where the group should be
    split and test if the split is correct. This effectively yields two consecutive groups.

    For example "on the internet" is phonemized into "ɔnðɪ ɪntɚnɛt", but "on the" becomes "ɔnðə", which is not the
    same as "ɔnðɪ".
    On our first attempt, we map [on] to [ɔnðɪ], and test if [the internet] becomes [ɪntɚnɛt]. It fails.
    On our second attempt, we map [on the] to [ɔnðɪ], and test if [internet] becomes [ɪntɚnɛt]. It passes, so we
    know that this grouping is correct.

    :return:
        - mapping: the mapping updated with the first group found
        - text_range: the number of text words in the second group
        - conv_range: the number of converted words in the second group
    """
    # Copy the mapping
    mapping = list(mapping)

    # Iterate over all the subgrouping possibilities for the given group
    for first_group in _backtracking_group_generator(text_range, conv_range):
        # Create a temporary mapping with the addition of the first group
        sub_mapping = mapping + [(mapping[-1][0] + first_group[0], mapping[-1][1] + first_group[1])]

        # Perform a sweep search, disallowing the use of any previous context. This ensures that the function will
        # succeed in finding a group only if the mapping given is accurate.
        second_group = _wl_sweep_search(sub_mapping, text_words, conv_words, conv_fn, eq_fn, max_prev_groups=0)

        # If the sweep search succeeds, the first group was correctly guessed and the sweep returned the second.
        if second_group[0]:
            return (sub_mapping, *second_group)

    # We couldn't improve the mapping somehow, we return the suboptimal group
    return mapping + [(mapping[-1][0] + text_range, mapping[-1][1] + conv_range)], None, None
