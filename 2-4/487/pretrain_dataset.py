from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.utils import read_strings
from utils.preprocess import preprocess_noisy
from noising import Noiser


def get_pretrain_data(data_path=None,
                      corpus=None,
                      val_ratio=0.05,
                      add_spacing=False,
                      logger=None,
                      use_corpus=True):

    if corpus is None:
        corpus = read_strings(data_path)
    noiser = Noiser()
    if add_spacing:
        from pykospacing import Spacing
        spacing = Spacing()
    corpus = [preprocess_noisy(sent) for sent in corpus]
    pairs = []
    for idx, sent in enumerate(corpus):
        if len(sent) == 0:
            continue
        pair = {}
        # noisy
        pair['noisy'] = noiser.noise(sent, corpus=corpus if use_corpus else None)
        # clean
        clean_sent = sent
        if add_spacing:
            clean_sent = spacing(clean_sent)
        pair['clean'] = clean_sent
        pairs.append(pair)

        if idx % 10000 == 0:
            log_fn = logger.info if logger is not None else print
            log_fn(f'preparing data: {idx} / {len(corpus)}')

    if val_ratio == 0:
        return pairs

    train_data, valid_data = train_test_split(pairs, test_size=val_ratio)

    return train_data, valid_data


def get_pretrain_dataloader(args, data, tokenizer, mode, do_multitask=False, drop=0.0):
    del do_multitask, drop  # unused
    dataset = PretrainDataset(data, mode, tokenizer)
    batch_size = args.train_batch_size if mode == 'train' else args.eval_batch_size
    dataloader = DataLoader(dataset,
                            shuffle=mode == 'train',
                            batch_size=batch_size,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn)

    return dataloader


class PretrainDataset(Dataset):

    def __init__(self, data, mode, tokenizer):
        self.data = data
        self.mode = mode

        self.tokenizer = tokenizer
        self.unk_idx = tokenizer.unk_idx
        self.pad_idx = tokenizer.pad_idx
        self.sos_idx = tokenizer.sos_idx
        self.eos_idx = tokenizer.eos_idx
        self.cls_idx = tokenizer.cls_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        source_text = [x['noisy'] for x in data]
        target_text = [x['clean'] for x in data]
        src_padded, src_padding_mask = self.tokenizer(source_text)
        tgt_padded, tgt_padding_mask = self.tokenizer(target_text, is_target=True)

        return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask
