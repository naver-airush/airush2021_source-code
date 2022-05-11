import argparse

import os
import json # import json module

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

from data_loader import read_strings

import nsml
from nsml import DATASET_PATH

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument("--vocab_size", type=int, default=6000) # 만들 Vocab의 숫자 
    parser.add_argument("--limit_alphabet", type=int, default=6000)

    args = parser.parse_args()
    return args

def postprocess_state(sentence : str) -> str:
    """TRADE, SUMBT postprocessing
    Args:
        state (List[str]): state prediction
    Returns:
        List[str]: postprocessing state
    """
    sentence = sentence.replace(" : ", ":").replace(" , ", ", ").replace('( ', '(').replace(' )', ')').replace(' & ', '&').replace(' = ', '=')
    sentence = sentence.replace(" % ", "%").replace(' ~ ', '~').replace(' ^ ', '^')
    if sentence.endswith(' ~'):
        sentence = sentence.replace(' ~', '~')
    if sentence.endswith(' ^^'):
        sentence = sentence.replace(' ^^', '^^')
    if sentence.endswith(' ^'):
        sentence = sentence.replace(' ^', '^')
    if sentence.endswith('......'):
        sentence = sentence.replace('......', ' ......')
    sentence = sentence.replace(') 에', ')에').replace('곳 (', '곳(').replace('부터~트', '부터~ 트').replace('# 정왕동', '#정왕동')
    sentence = sentence.replace('쨘 -', '쨘-').replace('해드리겠습니다!', '해드리겠습니다 !').replace('6 / 6', '6/6').replace('6 / 4', '6/4')
    sentence = sentence.replace('> ㅋ', '>ㅋ').replace('이상~헤', '이상~ 헤').replace('6 / 6', '6/6').replace('6 / 4', '6/4')

    return sentence

def main():
    args = get_args()

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False, # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##"
    )

    noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
    annotations = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
    corpuses = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
    clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

    corpus = noisy_sents + clean_sents + corpuses
    print(len(corpus))
    print(len(list(set(corpus))))

    tokenizer.train_from_iterator(
        corpus,
        limit_alphabet=args.limit_alphabet,
        vocab_size=args.vocab_size
    )

    vocab_path = f"custom_{args.limit_alphabet}_{args.vocab_size}_tokenizer"
    tokenizer.save(vocab_path, True)

    vocab_file = f"custom_{args.limit_alphabet}_{args.vocab_size}_tokenizer.txt"
    f = open(vocab_file,'w',encoding='utf-8')
    with open(vocab_path) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')

        f.close()

    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)

    # print(f"vocab size is {tokenizer.vocab_size}")
    # print('-' * 50)

    # for i, string in enumerate(corpus):
    #     postprocess_string = postprocess_state(tokenizer.decode([tok for tok in tokenizer.encode(string) if tok >= 4]))
    #     if string != postprocess_string:
    #         if tokenizer.encode(postprocess_string) != tokenizer.encode(string):
    #             print(f"[바꾼] {postprocess_string}")
    #             print(f"[이전] {string}")
    #             print(f"[인코딩바꾼] {tokenizer.encode(postprocess_string)}")
    #             print(f"[인코딩이전] {tokenizer.encode(string)}")
    #             print()
                
    #     if not i % 1000:
    #         print(i)

if __name__ == "__main__":
    main()