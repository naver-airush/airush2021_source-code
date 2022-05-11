import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data, tokenizer, max_seq_length=None):
    src_token_ids = [tokenizer(x['noisy']) for x in data]
    tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()
    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


#%% smart batch

import random

def make_smart_batches(data, tokenizer, batch_size, anno_label=False, padding_token_id=1):
        
    # SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

    src_token_ids = [tokenizer(x['noisy']) for x in data]
    tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]
    
    indexs = list(range(len(data)))
    
    if anno_label :
        anno_labels = [x['annotation'] for x in data]
    else :
        anno_labels = None

    if anno_labels :
        samples = sorted(zip(src_token_ids, tgt_token_ids, indexs, anno_labels), key=lambda x: len(x[0]))
    else :
        samples = sorted(zip(src_token_ids, tgt_token_ids, indexs), key=lambda x: len(x[0]))
    
    ordered_index = []
    batch_ordered_src = []
    batch_ordered_tgt = []
    if anno_labels :
        batch_ordered_anno = []

    while len(samples) > 0:
        
        to_take = min(batch_size, len(samples))
        select = random.randint(0, len(samples) - to_take)
        batch = samples[select:(select + to_take)]

        batch_ordered_src.append([s[0] for s in batch])
        batch_ordered_tgt.append([s[1] for s in batch])
        
        for s in batch :
            ordered_index.append(s[2])
        
        if anno_labels :
            batch_ordered_anno.append(torch.tensor([s[3] for s in batch]))

        del samples[select:select + to_take]

    src_padded = []; src_padding_mask = []
    tgt_padded_input = []; tgt_padding_mask = []; tgt_padded_output = []
    for batch_inputs_src, batch_inputs_tgt in zip(batch_ordered_src, batch_ordered_tgt) :
        batch_padded_src = []; batch_padded_tgt = []
        batch_mask_src = []; batch_mask_tgt = []

        max_size_src = max(max([len(sen) for sen in batch_inputs_src]), 5)
        max_size_tgt = max([len(sen) for sen in batch_inputs_tgt])

        for sen_src, sen_tgt in zip(batch_inputs_src, batch_inputs_tgt) :
            num_pads_src = max_size_src - len(sen_src)
            num_pads_tgt = max_size_tgt - len(sen_tgt)

            padded_input_src = sen_src + [padding_token_id]*num_pads_src
            padded_input_tgt = sen_tgt + [padding_token_id]*num_pads_tgt
            mask_src = [1] * len(sen_src) + [0] * num_pads_src
            mask_tgt = [1] * (len(sen_tgt)-1) + [0] * num_pads_tgt

            batch_padded_src.append(padded_input_src)
            batch_padded_tgt.append(padded_input_tgt)
            batch_mask_src.append(mask_src)
            batch_mask_tgt.append(mask_tgt)

        batch_padded_src = torch.tensor(batch_padded_src).t().contiguous()
        batch_mask_src = torch.tensor(batch_mask_src).bool().t()
        batch_padded_tgt = torch.tensor(batch_padded_tgt).t().contiguous()
        batch_mask_tgt = torch.tensor(batch_mask_tgt).bool().t()

        src_padded.append(batch_padded_src)
        tgt_padded_input.append(batch_padded_tgt[:-1])
        src_padding_mask.append(batch_mask_src)
        tgt_padding_mask.append(batch_mask_tgt)
        tgt_padded_output.append(batch_padded_tgt[1:])

    if anno_labels :
        train = (src_padded, tgt_padded_input, src_padding_mask, tgt_padding_mask, tgt_padded_output, batch_ordered_anno), ordered_index
    else :
        train = (src_padded, tgt_padded_input, src_padding_mask, tgt_padding_mask, tgt_padded_output), ordered_index

    return train


#%%
def make_smart_batches_pretrain(data, tokenizer, batch_size, padding_token_id=1):
        
    # SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']
    
    token_ids = [[2] + tokenizer(x) + [3] for x in data]

    samples = sorted(token_ids, key=lambda x: len(x))
    
    batch_ordered_src = []
    while len(samples) > 0:        
        to_take = min(batch_size, len(samples))
        select = random.randint(0, len(samples) - to_take)
        batch = samples[select:(select + to_take)]
        batch_ordered_src.append([s for s in batch])
        del samples[select:select + to_take]

    padded_input = []; padding_mask = []; padded_output = []
    for batch_inputs_src in batch_ordered_src :
        batch_padded_src = []; batch_mask_src = []

        max_size_src = max(max([len(sen) for sen in batch_inputs_src]), 3)

        for sen_src in batch_inputs_src :
            num_pads_src = max_size_src - len(sen_src)

            padded_input_src = sen_src + [padding_token_id]*num_pads_src
            mask_src = [1] * (len(sen_src)-1) + [0] * num_pads_src

            batch_padded_src.append(padded_input_src)
            batch_mask_src.append(mask_src)

        batch_padded_src = torch.tensor(batch_padded_src).T.contiguous()
        batch_mask_src = torch.tensor(batch_mask_src).T.bool()

        padded_input.append(batch_padded_src[:-1])
        padding_mask.append(batch_mask_src)
        padded_output.append(batch_padded_src[1:].T)

    train = (padded_input, padding_mask, padded_output)

    return train


