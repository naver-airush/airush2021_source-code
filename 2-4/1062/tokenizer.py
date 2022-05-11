from collections import Counter, defaultdict

from data_loader import read_strings, write_strings

SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


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
        # print(len(char_counter)) # 2366
        i2c = SPECIAL_TOKENS
        i2c += [c for c, _ in char_counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
        return cls(i2c)

    def save(self, path):
        write_strings(path, self.i2c)

    def load(self, path):
        i2c = read_strings(path)
        self.init(i2c)
