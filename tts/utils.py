import torch
from munch import Munch

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text, dataset):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(char, dataset, text)
        return indexes


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def inference(model, tokens, ref_s, ref_p, _prsinf, _boundaries, device):
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        _boundaries = torch.from_numpy(_boundaries).to(device)
        _prsinf = torch.from_numpy(_prsinf).to(device)
        starts, ends = _boundaries[0], _boundaries[1]-1
        assert (ends > starts-1).all()
        num_phon = ends - starts + 1
        prsinf_d = _prsinf[:, 0].repeat_interleave(num_phon, dim=0)
        dur_wd_emb = model.embed_dur(prsinf_d)
        d_en = d_en + dur_wd_emb.transpose(-1, -2)
        
        d = model.predictor.text_encoder(d_en, 
                                         ref_p, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        num_sil = int(pred_dur[-2:].sum())

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        prefix_sum = pred_aln_trg.int().sum(dim=-1).cumsum(dim=0).to(device)
        sums = torch.where(
            starts == 0,
            prefix_sum[ends],
            prefix_sum[ends] - prefix_sum[starts - 1]
        )
        prsinf_p = _prsinf[:, 1:].repeat_interleave(sums, dim=0)
        if not prsinf_p.size(0) == prefix_sum[-1]:
            set_trace()
        
        f0N_emb = model.embed_f0N(prsinf_p.transpose(0,1).unsqueeze(0)).transpose(-1, -2)
        f0N_emb = f0N_emb.reshape(f0N_emb.size(0), -1, f0N_emb.size(-1))
        
        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        
        asr_new = torch.zeros_like(en)
        asr_new[:, :, 0] = en[:, :, 0]
        asr_new[:, :, 1:] = en[:, :, 0:-1]
        en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en+f0N_emb, ref_p)
        F0_pred[0, -num_sil:] = 0
        N_pred[0, -num_sil:] = -6

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        
        asr_new = torch.zeros_like(asr)
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref_s.squeeze().unsqueeze(0))
    
    return out.squeeze().cpu().numpy()[..., :-50]