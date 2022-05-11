import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.utils import read_strings
from utils.preprocess import preprocess_noisy

ANNOTATIONS = [
    'perfect', 'spacing', 'pasting', 'tense', 'honorific', 'punctuation', 'typo',
    'advanced'
]


def get_train_val_data(data_path,
                       label_path,
                       annotation_path=None,
                       corpus_path=None,
                       val_ratio=0.05,
                       add_spacing=False,
                       add_pseudo_data=False,
                       mecab_normalize=False,
                       preprocess=False):
    noisy_sents = read_strings(data_path)
    if preprocess:
        noisy_sents = [preprocess_noisy(sent) for sent in noisy_sents]
    if mecab_normalize:
        import mecab
        mecab_tokenizer = mecab.MeCab().morphs
        noisy_sents = [' '.join(mecab_tokenizer(sent)) for sent in noisy_sents]
    if add_spacing:
        from pykospacing import Spacing
        spacing = Spacing()
        noisy_sents = [spacing(sent) for sent in noisy_sents]
    clean_sents = read_strings(label_path)
    if annotation_path is not None:
        annot_sents = read_strings(annotation_path)
        pairs = [{
            'noisy': noisy,
            'clean': clean,
            'annotation': annotation
        } for noisy, clean, annotation in zip(noisy_sents, clean_sents, annot_sents)]
    else:
        pairs = [{
            'noisy': noisy,
            'clean': clean
        } for noisy, clean in zip(noisy_sents, clean_sents)]

    if val_ratio == 0:
        train_data, valid_data = pairs, None
    else:
        train_data, valid_data = train_test_split(pairs, test_size=val_ratio)

    if add_pseudo_data:
        from noising import Noiser
        noiser = Noiser()
        pseudo_data = []
        for data in train_data:
            pseudo = noiser.noise(data['clean'])
            pseudo_data.append({'noisy': pseudo, 'clean': data['clean']})

        train_data += pseudo_data

    if corpus_path is not None:
        corpus_data = read_strings(corpus_path)
        corpus_data = [preprocess_noisy(sent) for sent in corpus_data]
        return train_data, valid_data, corpus_data
    return train_data, valid_data, None


def get_dataloader(args, data, tokenizer, mode, do_multitask=False, drop=0.0):
    dataset = TextDataset(data, mode, tokenizer, do_multitask=do_multitask, drop=drop)
    if mode == 'train':
        batch_size = args.train_batch_size
    elif mode == 'unlabeled':
        batch_size = args.unlabeled_batch_size
    else:
        batch_size = args.eval_batch_size

    dataloader = DataLoader(dataset,
                            shuffle=mode in ('train', 'unlabeled'),
                            batch_size=batch_size,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)

    return dataloader


class TrainValDataset:

    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    def save(self, path):
        with open(os.path.join(path, 'train'), 'w', encoding='utf-8') as train_f:
            json.dump(self.train_data, train_f)
        with open(os.path.join(path, 'val'), 'w', encoding='utf-8') as val_f:
            json.dump(self.val_data, val_f)

    def load(self, path):
        with open(os.path.join(path, 'train'), 'r', encoding='utf-8') as train_f:
            self.train_data = json.load(train_f)
        with open(os.path.join(path, 'val'), 'r', encoding='utf-8') as val_f:
            self.val_data = json.load(val_f)


class TextDataset(Dataset):

    def __init__(self, data, mode, tokenizer, do_multitask=False, drop=0.0):
        self.data = data
        self.mode = mode
        self.do_multitask = do_multitask
        self.drop = drop

        self.tokenizer = tokenizer
        self.unk_idx = tokenizer.unk_idx
        self.pad_idx = tokenizer.pad_idx
        self.sos_idx = tokenizer.sos_idx
        self.eos_idx = tokenizer.eos_idx
        self.cls_idx = tokenizer.cls_idx
        self.mask_idx = tokenizer.mask_idx

        self.annotation_dict = {annot: idx for idx, annot in enumerate(ANNOTATIONS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        if self.mode in ('train', 'val'):
            source_text = [x['noisy'] for x in data]
            target_text = [x['clean'] for x in data]
            src_padded, src_padding_mask = self.tokenizer(
                source_text, drop=self.drop if self.mode == 'train' else 0.0)
            tgt_padded, tgt_padding_mask = self.tokenizer(
                target_text, is_target=True, add_special_tokens=self.do_multitask)

            if self.do_multitask:
                # annotation labels
                annot_label = []
                for x in data:
                    annot_idx = [
                        self.annotation_dict[annot]
                        for annot in x['annotation'].split(',')
                    ]
                    _label = torch.zeros(len(self.annotation_dict))
                    for i in annot_idx:
                        _label[i] = 1
                    annot_label.append(_label)

                annot_label = torch.stack(annot_label, dim=0)  # [bs, 8]

                return (src_padded, tgt_padded, src_padding_mask, tgt_padding_mask,
                        annot_label)

            return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

        src_padded, src_padding_mask = self.tokenizer(data)
        return src_padded, src_padding_mask
