import re
from data.utils import word_mappings


def build_pattern(tokenizer,
                  input_sent,
                  n_free_after=5,
                  free_char='@', 
                  sil='<SIL>',   
                  sep='[SEP_2]'):          

    input_sent_no_punct = re.sub(r"[^\w\s]", "", input_sent)
    words, _, phon_groups = word_mappings('test', input_sent_no_punct)
    
    parts = []
    for w in words:
        parts.append(f"{sil}{free_char}{w}{free_char * n_free_after}")
    
    parts.append(f"{sil}{free_char}{sep}")
    template = ''.join(parts)

    pattern = []
    for part in re.split(r'([@#]+)', template):
        if part.startswith('@'):
            pattern += [None] * len(part)  
        elif part:                                           
            pattern += tokenizer.encode(part, add_special_tokens=False)
    return template, pattern

