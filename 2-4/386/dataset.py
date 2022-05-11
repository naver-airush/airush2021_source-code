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
        src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)

        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [tokenizer.pad_token_id] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()
    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


def collate_fn_pretrain(data, tokenizer, max_seq_length=None):
    src_token_info = tokenizer([x['noisy'] for x in data])
    tgt_token_info = tokenizer([x['clean'] for x in data])

    src_max_seq_length = max([len(x) for x in src_token_info.input_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_info.input_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_info.input_ids, tgt_token_info.input_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)

        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [tokenizer.pad_token_id] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()
    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


def collate_fn_transformer(data, tokenizer, max_seq_length=None):
    # src_token_info = tokenizer([x['noisy'] for x in data])
    # tgt_token_info = tokenizer([x['clean'] for x in data])
    # src_token_ids = [tokenizer(x['noisy']) for x in data]
    src_token_ids = [[2] + tokenizer(x['noisy']) + [3] for x in data]
    tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    tgt_padded = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)

        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [tokenizer.pad_token_id] * tgt_pad_length)

    # src_padded = []
    # tgt_padded = []
    # for src, tgt in zip(src_token_info.input_ids, tgt_token_info.input_ids):
    #     src = src[:src_max_seq_length]
    #     src_pad_length = src_max_seq_length - len(src)
    #     src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)

    #     tgt = tgt[:tgt_max_seq_length]
    #     tgt_pad_length = tgt_max_seq_length - len(tgt)
    #     tgt_padded.append(tgt + [tokenizer.pad_token_id] * tgt_pad_length)

    src_padded = torch.LongTensor(src_padded).contiguous()
    src_padding_mask = src_padded.ne(tokenizer.pad_token_id)

    tgt_padded = torch.LongTensor(tgt_padded).contiguous()
    tgt_padding_mask = tgt_padded[:,:-1].ne(tokenizer.pad_token_id)
    return src_padded, tgt_padded[:,:-1], src_padding_mask, tgt_padding_mask, tgt_padded[:,1:]


def collate_fn_pretrain_transformer(data, tokenizer, max_seq_length=None):
    src_token_info = tokenizer([x['noisy'] for x in data])
    tgt_token_info = tokenizer([x['clean'] for x in data])

    src_max_seq_length = max([len(x) for x in src_token_info.input_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_info.input_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    tgt_padded = []
    for src, tgt in zip(src_token_info.input_ids, tgt_token_info.input_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)

        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [tokenizer.pad_token_id] * tgt_pad_length)

    src_padded = torch.LongTensor(src_padded).contiguous()
    src_padding_mask = src_padded.ne(tokenizer.pad_token_id)

    tgt_padded = torch.LongTensor(tgt_padded).contiguous()
    tgt_padding_mask = tgt_padded[:,:-1].ne(tokenizer.pad_token_id)
    return src_padded, tgt_padded[:,:-1], src_padding_mask, tgt_padding_mask, tgt_padded[:,1:]

def collate_fn_mlm_pretrain_transformer(data, tokenizer, max_seq_length=None):
    src_token_info = tokenizer([x for x in data])

    src_max_seq_length = max([len(x) for x in src_token_info.input_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length

    src_padded = []
    for src in src_token_info.input_ids:
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [tokenizer.pad_token_id] * src_pad_length)

    src_padded = torch.LongTensor(src_padded).contiguous()
    src_padding_mask = src_padded.ne(tokenizer.pad_token_id)

    return src_padded, torch.LongTensor([]), src_padding_mask, torch.LongTensor([]), torch.LongTensor([])
