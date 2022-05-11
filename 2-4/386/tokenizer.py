import os
import json # import json module

from collections import Counter, defaultdict

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

from data_loader import read_strings, write_strings

SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>', '<mask>']


class CharTokenizer(object):
    def __init__(self, i2c):
        self.init(i2c)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, sent):
        return [self.vocab[c] for c in sent]

    def init(self, i2c):
        self.i2c = i2c
        self.vocab = defaultdict(int)
        self.vocab.update({c: i for i, c in enumerate(i2c)})

    @classmethod
    def from_strings(cls, strings, vocab_size):
        char_counter = Counter()
        for x in strings:
            char_counter.update(x)

        print(f"character의 갯수 : {len(char_counter)}")
        
        i2c = SPECIAL_TOKENS
        i2c += [c for c, _ in char_counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
        return cls(i2c)

    @property
    def pad_token_id(self):
        return SPECIAL_TOKENS.index('<pad>')

    def convert_tokens_to_ids(self, item):
        return self.i2c.index(item)

    def save(self, path):
        write_strings(path, self.i2c)

    def load(self, path):
        i2c = read_strings(path)
        self.init(i2c)



class CustomTokenizer(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = None

    def __call__(self, sent):
        return self.tokenizer(sent)

    def __len__(self):
        return len(self.tokenizer)

    def get_tokenizer(self):
        self.tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False, # Must be False if cased model
            lowercase=False,
            wordpieces_prefix="##"
        )

        noisy_sents = read_strings(os.path.join(self.args.data_dir, "train_data", "train_data"))
        corpuses = read_strings(os.path.join(self.args.data_dir, "train_data", "train_corpus"))
        clean_sents = read_strings(os.path.join(self.args.data_dir, "train_label"))

        corpus = noisy_sents + clean_sents + corpuses

        self.tokenizer.train_from_iterator(
            corpus,
            limit_alphabet=self.args.limit_alphabet,
            vocab_size=self.args.tokenizer_vocab_size
        )

        vocab_path = f"custom_tokenizer"
        self.tokenizer.save(vocab_path, True)

        vocab_file = f"custom_tokenizer.txt"
        f = open(vocab_file,'w',encoding='utf-8')
        with open(vocab_path) as json_file:
            json_data = json.load(json_file)
            for item in json_data["model"]["vocab"].keys():
                f.write(item+'\n')

            f.close()

        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id if self.tokenizer is not None else 0

    def convert_tokens_to_ids(self, item):
        return self.tokenizer.convert_tokens_to_ids(item)

    def decode(self, sent):
        return self.tokenizer.decode(sent)

    def save(self, path):
        self.tokenizer.save_vocabulary(path)

    def load(self, path):
        self.tokenizer = BertTokenizer(vocab_file=path)





# def customtokenizer(args):
#     tokenizer = BertWordPieceTokenizer(
#         clean_text=True,
#         handle_chinese_chars=True,
#         strip_accents=False, # Must be False if cased model
#         lowercase=False,
#         wordpieces_prefix="##"
#     )

#     noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
#     corpuses = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
#     clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

#     corpus = noisy_sents + clean_sents + corpuses

#     tokenizer.train_from_iterator(
#         corpus,
#         limit_alphabet=args.limit_alphabet,
#         vocab_size=args.tokenizer_vocab_size
#     )

#     vocab_path = f"custom_tokenizer"
#     tokenizer.save(vocab_path, True)

#     vocab_file = f"custom_tokenizer.txt"
#     f = open(vocab_file,'w',encoding='utf-8')
#     with open(vocab_path) as json_file:
#         json_data = json.load(json_file)
#         for item in json_data["model"]["vocab"].keys():
#             f.write(item+'\n')

#         f.close()

#     tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
#     return tokenizer
