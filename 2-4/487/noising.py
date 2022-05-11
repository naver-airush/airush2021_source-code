from g2pk import G2p
import mecab
import random


class Noiser:

    def __init__(self):

        self.g2p = G2p()
        self.mecab_tokenizer = mecab.MeCab().morphs

    def noise(self, sent, corpus=None, p=0.1):
        sent = self.grapheme(sent)
        sent = self.add_spacing_noise(sent, p=p)
        if corpus is not None:
            sent = self.delete_token(sent, p=p)
            sent = self.add_token(sent, corpus, p=p)
            sent = self.replace_token(sent, corpus, p=p)
        return sent

    def grapheme(self, sent):
        return self.g2p(sent, group_vowels=True)

    def add_spacing_noise(self, sent, p=0.1):
        tokenized = ' '.join(self.mecab_tokenizer(sent))
        noised = []
        for char in tokenized:
            if char == ' ' and random.randint(0, 1) < p:
                continue
            noised.append(char)
        return ''.join(noised)

    def identity(self, sent):
        return sent

    def delete_token(self, sent, p=0.1):
        noised = []
        for char in sent:
            if random.randint(0, 1) < p:
                continue
            noised.append(char)
        return ''.join(noised)

    def add_token(self, sent, corpus, p=0.1):
        noised = []
        for char in sent:
            if random.randint(0, 1) < p:
                random_sent = random.choice(corpus)
                while len(random_sent) == 0:
                    random_sent = random.choice(corpus)
                random_tok = random.choice(random_sent)
                noised.append(random_tok)
            noised.append(char)
        return ''.join(noised)

    def replace_token(self, sent, corpus, p=0.1):
        noised = []
        for char in sent:
            if random.randint(0, 1) < p:
                random_sent = random.choice(corpus)
                while len(random_sent) == 0:
                    random_sent = random.choice(corpus)
                random_tok = random.choice(random_sent)
                noised.append(random_tok)
            else:
                noised.append(char)
        return ''.join(noised)

    def heterograph_noise(self, sent):
        pass
