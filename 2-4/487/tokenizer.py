import os
from collections import Counter
import random

from jamo import h2j
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import ElectraTokenizerFast
from tokenizers import Tokenizer, SentencePieceBPETokenizer, normalizers
from tokenizers.processors import TemplateProcessing

from utils.utils import read_strings, write_strings, reconstruct_jamo

SPECIAL_TOKENS = ['[UNK]', '[PAD]', '[SOS]', '[EOS]', '[CLS]', '[SEP]', '[MASK]']


class CharTokenizer(object):

    def __init__(self, i2c, max_seq_length=None, use_jamo=False):
        self.max_len = max_seq_length
        self.use_jamo = use_jamo
        self.unk_idx = 0
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.cls_idx = 4
        self.sep_idx = 5
        self.mask_idx = 6
        self.init(i2c)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, sent, is_target=False, add_special_tokens=False, drop=0.0):
        if is_target:
            if add_special_tokens:
                token_ids = [
                    torch.LongTensor([self.sos_idx] + self.tokenize(x) + [self.eos_idx] +
                                     [self.cls_idx]) for x in sent
                ]
            else:
                token_ids = [
                    torch.LongTensor([self.sos_idx] + self.tokenize(x) + [self.eos_idx])
                    for x in sent
                ]
        else:
            token_ids = [torch.LongTensor(self.tokenize(x, drop=drop)) for x in sent]
        padded = pad_sequence(token_ids, batch_first=True,
                              padding_value=self.pad_idx)[:, :self.max_len]
        padding_mask = padded == self.pad_idx

        return padded, padding_mask

    def tokenize(self, sent, drop=0.0):
        if self.use_jamo:
            sent = h2j(sent)
        ids = []
        for c in sent:
            if drop > 0 and random.randint(0, 1) < drop:
                ids.append(self.mask_idx)
            else:
                ids.append(self.vocab.get(c, 0))
        return ids

    def decode(self, sent, skip_special_tokens=False):
        if isinstance(sent[0], list):
            decoded = self.batch_decode(sent, skip_special_tokens=skip_special_tokens)
        else:
            decoded = self.index_to_strings(sent, skip_special_tokens=skip_special_tokens)
        return decoded

    def index_to_strings(self, sent, skip_special_tokens=False):
        decoded = ''.join([
            self.i2c[tok]
            for tok in sent
            if ((tok > self.mask_idx) or not skip_special_tokens) and tok < len(self.i2c)
        ])
        if self.use_jamo:
            decoded = reconstruct_jamo(decoded)
        return decoded

    def batch_decode(self, batch_sent, skip_special_tokens=False):
        decoded = [
            self.index_to_strings(sent, skip_special_tokens=skip_special_tokens)
            for sent in batch_sent
        ]
        return decoded

    def init(self, i2c):
        self.i2c = i2c
        self.vocab = {c: i for i, c in enumerate(i2c)}

    @classmethod
    def from_strings(cls, strings, vocab_size, max_seq_length=None, use_jamo=False):
        char_counter = Counter()
        for x in strings:
            if use_jamo:
                x = h2j(x)
            char_counter.update(x)
        i2c = SPECIAL_TOKENS
        i2c += [c for c, _ in char_counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
        return cls(i2c, max_seq_length=max_seq_length, use_jamo=use_jamo)

    def save(self, path):
        path = os.path.join(path, 'vocab.txt')
        write_strings(path, self.i2c)

    def load(self, path):
        path = os.path.join(path, 'vocab.txt')
        i2c = read_strings(path)
        self.init(i2c)


class HFTokenzer:
    """HuggingFace transformers Tokenizers.
    """

    def __init__(self, args):
        self.max_len = args.max_seq_length

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, sentences, is_target=False, add_special_tokens=False, drop=0.2):
        if is_target:
            sentences = [
                self.tokenizer.bos_token + sent + self.tokenizer.eos_token
                for sent in sentences
            ]
        encoded = self.tokenizer(sentences,
                                 add_special_tokens=add_special_tokens and not is_target,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 return_token_type_ids=False,
                                 truncation=True,
                                 return_tensors='pt')
        if isinstance(sentences, str):
            encoded = {key: encoded[key][0] for key in encoded}

        padded = encoded['input_ids']
        padding_mask = encoded['attention_mask']
        padding_mask = (1 - padding_mask).bool()
        return padded, padding_mask

    def decode(self, sent, skip_special_tokens=False):
        if isinstance(sent[0], list):
            decoded = self.tokenizer.batch_decode(sent,
                                                  skip_special_tokens=skip_special_tokens)
        else:
            decoded = self.tokenizer.decode(sent, skip_special_tokens=skip_special_tokens)
        return decoded

    def save(self, path):
        pass

    def load(self, path):
        self.tokenizer = ElectraTokenizerFast.from_pretrained(path)
        self.tokenizer.add_special_tokens({'bos_token': '[SOS]', 'eos_token': '[EOS]'})
        self.unk_idx = self.tokenizer.unk_token_id
        self.pad_idx = self.tokenizer.pad_token_id
        self.sos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.sep_idx = self.tokenizer.sep_token_id
        self.mask_idx = self.tokenizer.mask_token_id


class SentencePienceTrainer:

    def __init__(self, args):
        self.vocab_size = args.vocab_size  # 30000
        self.use_jamo = args.use_jamo
        self.tokenizer = SentencePieceBPETokenizer()
        if self.use_jamo:
            self.tokenizer.normalizer = normalizers.NFKD()
        self.tokenizer.post_processor = TemplateProcessing(special_tokens=[
            (tok, i) for i, tok in enumerate(SPECIAL_TOKENS)
        ],)
        self.limit_alphabet = 6000
        self.min_frequency = 1

        ## 1) define special tokens
        unused_list = [f'[unused{n}]' for n in range(100)]
        self.user_defined_symbols = SPECIAL_TOKENS + unused_list

    def mecab_tokenize(self, train_sents):
        import mecab
        mecab_tokenizer = mecab.MeCab().morphs
        print('mecab check :', mecab_tokenizer('어릴때보고 지금다시봐도 재밌어요ㅋㅋ'))
        total_morph = [' '.join(mecab_tokenizer(sent)) for sent in train_sents]

        return total_morph

    def word_piece_train(self, files=None, iterator=None):
        if files is None:
            assert iterator is not None
            self.tokenizer.train_from_iterator(
                iterator=iterator,
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                limit_alphabet=self.limit_alphabet,
                show_progress=True,
                special_tokens=self.user_defined_symbols,
            )
        else:
            self.tokenizer.train(
                files=files,
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                limit_alphabet=self.limit_alphabet,
                show_progress=True,
                special_tokens=self.user_defined_symbols,
            )

        print('train complete')
        print(self.tokenizer.get_vocab())

        sentence = '나는 오늘 아침밥을 먹었다.'
        output = self.tokenizer.encode(sentence)
        print(sentence)
        print('=>idx   : %s' % output.ids)
        print('=>tokens: %s' % output.tokens)
        decoded = self.tokenizer.decode(output.ids)
        if self.use_jamo:
            decoded = reconstruct_jamo(decoded)
        print('=>decode: %s\n' % decoded)

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'tokenzer.tok')
        self.tokenizer.save(save_path)

        if self.use_jamo:
            checker = os.path.join(path, 'use_jamo.txt')
            with open(checker, 'w', encoding='utf-8') as f:
                f.write('True!')


class SPTokenizer:

    def __init__(self, args):
        self.max_len = args.max_seq_length

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, sentences, is_target=False, add_special_tokens=False, drop=0.2):
        if is_target:
            sentences = ['[SOS]' + sent + '[EOS]' for sent in sentences]
        encoded = self.tokenizer.encode_batch(sentences,
                                              add_special_tokens=add_special_tokens and
                                              not is_target)
        padded = torch.LongTensor([enc.ids for enc in encoded])
        padding_mask = torch.LongTensor([enc.attention_mask for enc in encoded])
        padding_mask = (1 - padding_mask).bool()
        return padded, padding_mask

    def decode(self, sent, skip_special_tokens=False):
        if isinstance(sent[0], list):
            decoded = self.tokenizer.decode_batch(sent,
                                                  skip_special_tokens=skip_special_tokens)
            if self.use_jamo:
                decoded = [reconstruct_jamo(dec) for dec in decoded]
        else:
            decoded = self.tokenizer.decode(sent, skip_special_tokens=skip_special_tokens)
            if self.use_jamo:
                decoded = reconstruct_jamo(decoded)
        return decoded

    def save(self, path):
        path = os.path.join(path, 'tokenzer.tok')
        self.tokenizer.save(path)

    def load(self, path):
        tok_path = os.path.join(path, 'tokenzer.tok')
        self.tokenizer = Tokenizer.from_file(tok_path)
        vocab = self.tokenizer.get_vocab()
        self.unk_idx = vocab['[UNK]']
        self.pad_idx = vocab['[PAD]']
        self.sos_idx = vocab['[SOS]']
        self.eos_idx = vocab['[EOS]']
        self.cls_idx = vocab['[CLS]']
        self.sep_idx = vocab['[SEP]']
        self.mask_idx = vocab['[MASK]']

        self.tokenizer.enable_padding(pad_id=self.pad_idx,
                                      pad_token='[PAD]',
                                      length=self.max_len)
        self.tokenizer.enable_truncation(self.max_len)

        if os.path.exists(os.path.join(path, 'use_jamo.txt')):
            self.use_jamo = True
        else:
            self.use_jamo = False
