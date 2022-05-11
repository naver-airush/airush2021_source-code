import argparse
from json import encoder
import logging

import os
import json
from copy import deepcopy
import numpy as np
import random

import torch
import torch.nn as nn

from data_loader import read_strings
from model import TransformerS2Model
from tokenizer import CharTokenizer
from evaluation import em, gleu

import nsml
from nsml import DATASET_PATH

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument("--val_ratio", type=float, default=0.2)
    # parser.add_argument("--num_val_data", type=int, default=1000)

    # tokenizer
    parser.add_argument("--limit_alphabet", type=int, default=6000)
    parser.add_argument("--tokenizer_vocab_size", type=int, default=6000)

    # model
    parser.add_argument("--model", type=str, default='transformer')
    parser.add_argument("--vocab_size", type=int, default=1600)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--ffn_size", type=int, default=2048)
    parser.add_argument("--share_decoder_input_output_embed", type=str2bool, default=False)

    # bert
    parser.add_argument("--bert", type=str, default='original')
    parser.add_argument("--bert_hidden_size", type=int, default=768)
    parser.add_argument("--bert_num_attention_heads", type=int, default=12)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--bert_intermediate_size", type=int, default=3072)

    # training
    parser.add_argument("--kfold", type=str2bool, default=False)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--mlm_pre", type=str2bool, default=False)
    parser.add_argument("--num_pretrain_epochs", type=int, default=5)

    parser.add_argument("--train_all", type=str2bool, default=False)

    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tokenizer", type=str, default='char')
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument("--scheduler", type=str, default='lambda')
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_warmup_steps", type=int, default=4000)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_workers', type=int, default=20)

    # nsml
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default="0")

    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def str2bool(v : str) -> bool:
    """convert string argument to boolean
    Args:
        v (str)
    Raises:
        argparse.ArgumentTypeError: [Boolean value expected]
    Returns:
        bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

        for idx in range(len(self.models)):
            for parameter in self.models[idx].parameters():
                parameter.requires_grad = False

        self.model_list = torch.nn.ModuleList(self.models)

def correct_ensemble(models, tokenizer, test_data, args):
    for model in models.models:
        model.eval()
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        batch = test_data[i:i + args.eval_batch_size]

        if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            # src_token_ids = [tokenizer(x) for x in batch]
            src_token_ids = [[2] + tokenizer(x) + [3] for x in batch]
        elif args.tokenizer in ['dsksd', 'plus', 'wordpiece']:
            src_token_ids = tokenizer(batch).input_ids

        src_seq_length = [len(x) for x in src_token_ids]
        src_max_seq_length = max(src_seq_length)

        src_padded = []
        for x in src_token_ids:
            x = x[:src_max_seq_length]
            src_pad_length = src_max_seq_length - len(x)
            src_padded.append(x + [tokenizer.pad_token_id] * src_pad_length)
        
        src_padded = torch.LongTensor(src_padded).contiguous().to(args.device)
        bert_encoder_padding_mask = src_padded.eq(tokenizer.pad_token_id)
        
        bert_encoder_outs = []
        encoder_outs = []
        for model in models.models:
            bert_encoder_out = model.bert_encoder(src_padded, attention_mask=~bert_encoder_padding_mask)
            bert_encoder_out = bert_encoder_out.last_hidden_state
            
            bert_encoder_out = bert_encoder_out.permute(1,0,2).contiguous()
            bert_encoder_out = {
                'bert_encoder_out': bert_encoder_out,
                'bert_encoder_padding_mask': bert_encoder_padding_mask,
            }
        
            encoder_out = model.encoder(src_padded, bert_encoder_out=bert_encoder_out)
            
            bert_encoder_outs.append(bert_encoder_out)
            encoder_outs.append(encoder_out)

        tgt_token_ids = [[2] for _ in batch]
        end = [False for _ in batch]
        for l in range(src_max_seq_length + 20):
            tgt_padded = torch.LongTensor(tgt_token_ids).contiguous().to(args.device)

            outputs = None
            for (model, bert_encoder_out, encoder_out) in zip(models.models, bert_encoder_outs, encoder_outs):
                output = model.decoder(tgt_padded, encoder_out=encoder_out, bert_encoder_out=bert_encoder_out)[0]

                if outputs is None:
                    outputs = output
                else:
                    outputs += output

            top1 = outputs[:,-1,:].argmax(-1).tolist()

            for i, tok in enumerate(top1):
                if tok == 3 or l >= src_seq_length[i] + 20:
                    end[i] = True
                tgt_token_ids[i].append(tok if not end[i] else 3)
            if all(end):
                break

        if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            # prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 5]) for x in tgt_token_ids])
            prediction.extend([postprocess_state(''.join([tokenizer.i2c[tok] for tok in x if tok >= 5])) for x in tgt_token_ids])
        elif args.tokenizer in ['dsksd', 'plus', 'wordpiece']:
            prediction.extend([postprocess_state(tokenizer.decode([tok for tok in x if tok >= 5])) for x in tgt_token_ids])
    return prediction


def bind_nsml(model, tokenizer=None, args=None, train_data=None):
    def save(path, **kwargs):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if tokenizer is not None and args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            tokenizer.save(os.path.join(path, 'vocab.txt'))

        valid_noisy = [x['noisy'] for x in train_data[1:200]]
        valid_clean = [x['clean'] for x in train_data[1:200]]

        prediction = correct_ensemble(model, tokenizer, valid_noisy, args)

        val_em = em(prediction, valid_clean)
        val_gleu = gleu(prediction, valid_clean)

        print(val_em, val_gleu)



    def load(path, **kwargs):
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        if tokenizer is not None and args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            tokenizer.load(os.path.join(path, 'vocab.txt'))


    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''
        return correct_ensemble(model, tokenizer, test_data, args)

    import nsml
    nsml.bind(save, load, infer)


def main():
    args = get_args()
    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
        tokenizer = CharTokenizer([])

    model = TransformerS2Model(tokenizer, args=args).to(args.device)

    # bind_nsml(model, tokenizer, args)

    if args.mode == "train":
        noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
        clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

        pairs = [{"noisy": noisy, "clean": clean} for noisy, clean in zip(noisy_sents, clean_sents)]

        if args.train_all:
            train_data = pairs
            valid_data = None
            logger.info(f"# of train data: {len(train_data)}")
        else:
            train_data, valid_data = pairs[:-int(len(pairs) * args.val_ratio)], pairs[-int(len(pairs) * args.val_ratio):]
            logger.info(f"# of train data: {len(train_data)}")
            logger.info(f"# of valid data: {len(valid_data)}")

        print(f"train_data is {train_data[0]}")

        train_sents = [x['noisy'] for x in train_data] + [x['clean'] for x in train_data]

        print(f"first train_sentences is {train_sents[0]}")
        print(f"last train_sentences is {train_sents[-1]}")

        if args.tokenizer == 'char':
            tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)



        bind_nsml(model, tokenizer, args, train_data)

        nsml.load(checkpoint='model_45', session='KR95329/airush2021-2-4/302')
        model1 = deepcopy(model)


        nsml.load(checkpoint='model_41', session='KR95329/airush2021-2-4/302')
        model2 = deepcopy(model)


        args.num_decoder_layers = 6
        model = TransformerS2Model(tokenizer, args=args).to(args.device)
        bind_nsml(model, tokenizer, args, train_data)


        nsml.load(checkpoint='model_34', session='KR95329/airush2021-2-4/296')
        model3 = deepcopy(model)


        args.num_encoder_layers = 12
        model = TransformerS2Model(tokenizer, args=args).to(args.device)
        bind_nsml(model, tokenizer, args, train_data)


        nsml.load(checkpoint='model_45', session='KR95329/airush2021-2-4/359')
        model4 = deepcopy(model)


        models = [model1, model2, model3, model4]#, model4, model5]

        model = Ensemble(models)
        model.to(args.device)

        bind_nsml(model, tokenizer, args, train_data)

        nsml.save('best')
    else:
        model1 = TransformerS2Model(tokenizer, args=args).to(args.device)
        model2 = TransformerS2Model(tokenizer, args=args).to(args.device)

        args.num_decoder_layers = 6
        model3 = TransformerS2Model(tokenizer, args=args).to(args.device)

        args.num_encoder_layers = 12
        model4 = TransformerS2Model(tokenizer, args=args).to(args.device)

        models = [model1, model2, model3, model4]#, model4, model5]

        model = Ensemble(models)
        model.to(args.device)

        bind_nsml(model, tokenizer, args)

    if args.pause:
        nsml.paused(scope=locals())


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, dim=1)

if __name__ == "__main__":
    main()