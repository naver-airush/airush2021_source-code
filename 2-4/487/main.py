import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from model import get_model, EnsembleModel, Ensemble3Model
from tokenizer import CharTokenizer, HFTokenzer, SPTokenizer, SentencePienceTrainer
from train import train, correct
from dataset import get_train_val_data, ANNOTATIONS, TrainValDataset
from pretrain_dataset import get_pretrain_data
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocess_noisy
from utils.utils import read_strings

import nsml
from nsml import DATASET_PATH

logging.basicConfig(format='%(asctime)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_dir',
                        type=str,
                        default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--add_pseudo_data', type=int, default=0)
    parser.add_argument('--token_dropout', type=float, default=0.0)
    parser.add_argument('--mecab_normalize', type=int, default=0)
    parser.add_argument('--preprocess', type=int, default=0)

    # model
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--intermediate_size', type=int, default=1024)
    parser.add_argument('--noise_scale', type=float, default=0.0)

    # training
    parser.add_argument('--ssl_mode',
                        type=str,
                        default='',
                        help='semi supervised learning',
                        choices=['pretrain', 'ssl'])
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--unlabeled_batch_size', type=int, default=64)

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        choices=['Adam', 'AdamW'])
    parser.add_argument('--scheduler',
                        type=str,
                        default='lambda',
                        choices=['lambda', 'linear'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--adam_betas', type=str, default='(0.9, 0.98)')
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--max_grad_norm', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--num_warmup_steps', type=int, default=4000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--turn_off_reg_step',
                        type=int,
                        default=1000,
                        help='turns off regularization by setting model.eval().'
                        '-1 means it is off.')

    parser.add_argument('--tokenizer',
                        type=str,
                        default='char',
                        choices=['char', 'sentencepiece', 'electra'])
    parser.add_argument('--use_mecab',
                        type=int,
                        default=1,
                        help='for sentencepiece train')
    parser.add_argument('--use_jamo', type=int, default=0, help='for char tokenizer')
    parser.add_argument('--tokenizer_session', type=str, default='93')
    parser.add_argument('--model',
                        type=str,
                        default='transformer',
                        choices=['transformer', 'bert_gpt', 'electra_gpt', 'ensemble'])
    parser.add_argument('--do_multitask', type=int, default=0)
    parser.add_argument('--share_embedding', type=int, default=1)
    parser.add_argument('--use_copy_attention', type=int, default=0)

    parser.add_argument('--ensemble_load_sessions', nargs='+', type=str, default='')
    parser.add_argument('--spacing_cache_session', type=str, default='')
    parser.add_argument('--load_model_session', type=str, default='')
    parser.add_argument('--data_session', type=str, default='')
    parser.add_argument('--data_type',
                        type=str,
                        default='pseudo',
                        choices=['pseudo', 'corpus'])
    parser.add_argument('--noise_type',
                        type=str,
                        default='direct',
                        choices=['direct', 'back_trans', 'all'])

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--n_gpu', type=int, default=1)

    # nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def bind_nsml(model, tokenizer=None, args=None):

    def save(path):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if tokenizer is not None and isinstance(tokenizer, CharTokenizer):
            tokenizer.save(path)

    def load(path):
        model.load_state_dict(
            torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                       map_location=lambda storage, loc: storage))
        if tokenizer is not None and isinstance(tokenizer, CharTokenizer):
            tokenizer.load(path)

    def infer(test_data):
        """
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        """
        test_data = [preprocess_noisy(sent) for sent in test_data]
        return correct(model, tokenizer, test_data, args)

    nsml.bind(save, load, infer)


def main():
    args = get_args()
    if args.tokenizer == 'sentencepiece':
        args.num_workers = 0
    logger.info(f'args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    if args.tokenizer == 'electra':
        tokenizer = HFTokenzer(args)
        tokenizer.load('monologg/koelectra-base-v3-discriminator')

    elif args.tokenizer == 'sentencepiece':
        tokenizer = SPTokenizer(args)
        nsml.bind(load=tokenizer.load)
        nsml.load(checkpoint='tokenizer',
                  session='KR95368/airush2021-2-4/' + args.tokenizer_session)

    elif args.tokenizer == 'char':
        tokenizer = CharTokenizer([],
                                  max_seq_length=args.max_seq_length,
                                  use_jamo=bool(args.use_jamo))
    if args.model == 'ensemble':
        model1 = get_model(args, tokenizer, annotation_size=len(ANNOTATIONS))
        model1 = model1.to(args.device)
        bind_nsml(model1, tokenizer, args)
        nsml.load(checkpoint='best',
                  session='KR95368/airush2021-2-4/' + args.ensemble_load_sessions[0])

        model2 = get_model(args, tokenizer, annotation_size=len(ANNOTATIONS))
        model2 = model2.to(args.device)
        bind_nsml(model2, tokenizer, args)
        nsml.load(checkpoint='best',
                  session='KR95368/airush2021-2-4/' + args.ensemble_load_sessions[1])

        if len(args.ensemble_load_sessions[1]) == 3:
            model3 = get_model(args, tokenizer, annotation_size=len(ANNOTATIONS))
            model3 = model3.to(args.device)
            bind_nsml(model3, tokenizer, args)
            nsml.load(checkpoint='best',
                      session='KR95368/airush2021-2-4/' + args.ensemble_load_sessions[2])
            model = Ensemble3Model(model1, model2, model3)
            model = model.to(args.device)
        else:
            model = EnsembleModel(model1, model2)
            model = model.to(args.device)

    else:
        model = get_model(args, tokenizer, annotation_size=len(ANNOTATIONS))
        model = model.to(args.device)
        logger.info(
            f'# of model parameters: {sum(p.numel() for p in model.parameters()) * 1e-6:.2f}M'
        )

    bind_nsml(model, tokenizer, args)
    if args.pause:
        nsml.paused(scope=locals())

    data_path = os.path.join(args.data_dir, 'train_data', 'train_data')
    label_path = os.path.join(args.data_dir, 'train_label')
    annotation_path = os.path.join(args.data_dir, 'train_data', 'train_annotation')
    corpus_path = os.path.join(args.data_dir, 'train_data', 'train_corpus')

    if args.mode == 'ensemble_save':
        nsml.save('best')
    elif args.mode == 'load_save':
        nsml.load(checkpoint='best',
                  session='KR95368/airush2021-2-4/' + args.load_model_session)
        nsml.save(args.load_model_session + '_best')

    elif args.mode in ('train', 'finetune'):
        if args.load_model_session:
            nsml.load(checkpoint='best',
                      session='KR95368/airush2021-2-4/' + args.load_model_session)
        if args.spacing_cache_session:
            dataset = TrainValDataset(None, None)
            nsml.bind(save=dataset.save, load=dataset.load)
            nsml.load(checkpoint='spacing_cache',
                      session='KR95368/airush2021-2-4/' + args.spacing_cache_session)
            train_data, valid_data = dataset.train_data, dataset.val_data
        elif args.data_session:
            # same valid set as pretrain...
            dataset = TrainValDataset(None, None)
            nsml.bind(save=dataset.save, load=dataset.load)
            nsml.load(checkpoint=args.data_type + '_data',
                      session='KR95368/airush2021-2-4/' + args.data_session)
            if args.noise_type != 'all':
                dataset.train_data = [
                    sent for sent in dataset.train_data if sent['type'] == args.noise_type
                ]
                dataset.val_data = [
                    sent for sent in dataset.val_data if sent['type'] == args.noise_type
                ]
            train_data = dataset.train_data if args.mode != 'finetune' else []
            valid_data = []
            ps_val_label = [data['clean'] for data in dataset.val_data]
            ps_val_label = set(ps_val_label)
            data, label = read_strings(data_path), read_strings(label_path)
            annot = read_strings(annotation_path)
            data = [preprocess_noisy(sent) for sent in data]
            for i, sent in enumerate(label):
                if sent in ps_val_label:
                    valid_data.append({
                        'noisy': data[i],
                        'clean': sent,
                        'annotation': annot[i]
                    })
                else:
                    train_data.append({
                        'noisy': data[i],
                        'clean': sent,
                        'annotation': annot[i]
                    })
            bind_nsml(model, tokenizer, args)
            corpus_data = read_strings(corpus_path) if args.ssl_mode else None
        else:
            train_data, valid_data, corpus_data = get_train_val_data(
                data_path,
                label_path,
                annotation_path=annotation_path if args.do_multitask else None,
                corpus_path=corpus_path if args.ssl_mode else None,
                val_ratio=args.val_ratio,
                add_pseudo_data=bool(args.add_pseudo_data),
                mecab_normalize=bool(args.mecab_normalize),
                preprocess=bool(args.preprocess))
        logger.info(f'# of train data: {len(train_data)}')
        if valid_data is not None:
            logger.info(f'# of valid data: {len(valid_data)}')
        else:
            logger.info(f'# of valid data: 0')
        if args.ssl_mode:
            logger.info(f'# of unlabeled data: {len(corpus_data)}')

        if isinstance(tokenizer, CharTokenizer) and not args.load_model_session:
            train_sents = [x['noisy'] for x in train_data] +\
                [x['clean'] for x in train_data]
            if args.ssl_mode:
                train_sents += corpus_data
            tokenizer = CharTokenizer.from_strings(train_sents,
                                                   args.vocab_size,
                                                   max_seq_length=args.max_seq_length,
                                                   use_jamo=args.use_jamo)
            print(tokenizer.vocab)
            bind_nsml(model, tokenizer, args)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, dim=1)

        train(model,
              tokenizer,
              train_data,
              valid_data,
              args,
              logger,
              unlabeled_data=corpus_data if args.ssl_mode == 'ssl' else None)

    elif args.mode == 'cache_data':
        data = read_strings(corpus_path if args.data_type == 'corpus' else label_path)
        if args.data_type == 'corpus':
            data = random.sample(data, 100000)
        train_data, val_data = train_test_split(data, test_size=args.val_ratio)
        if args.noise_type in ('back_trans', 'all'):
            # load back trans model
            nsml.load(checkpoint='best',
                      session='KR95368/airush2021-2-4/' + args.load_model_session)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model, dim=1)

        final_data = []
        for _data in [train_data, val_data]:
            back_trans, noised = [], []
            if args.noise_type in ('back_trans', 'all'):
                # back-trans
                back_trans = correct(model, tokenizer, _data, args, logger=logger)
                back_trans = [{
                    'noisy': preprocess_noisy(noisy),
                    'clean': clean,
                    'type': 'back_trans',
                } for noisy, clean in zip(back_trans, _data)]
            if args.noise_type in ('direct', 'all'):
                # direct noises
                noised = get_pretrain_data(corpus=_data,
                                           val_ratio=0,
                                           add_spacing=False,
                                           logger=logger,
                                           use_corpus=False)
                for sent in noised:
                    sent['type'] = 'direct'
            final_data.append(back_trans + noised)

        train_data, valid_data = final_data
        dataset = TrainValDataset(train_data, valid_data)
        nsml.bind(save=dataset.save, load=dataset.load)
        nsml.save(args.data_type + '_data')

        logger.info(f'# of train data: {len(train_data)}')
        logger.info(f'# of valid data: {len(valid_data)}')

    elif args.mode == 'load_save_cache':
        # same valid set as pretrain...
        dataset1 = TrainValDataset(None, None)  # direct
        nsml.bind(save=dataset1.save, load=dataset1.load)
        nsml.load(checkpoint='corpus_data', session='KR95368/airush2021-2-4/444')

        dataset2 = TrainValDataset(None, None)  # back_trans
        nsml.bind(save=dataset2.save, load=dataset2.load)
        nsml.load(checkpoint='corpus_data', session='KR95368/airush2021-2-4/446')
        # clean the cleans
        for _data in (dataset1.train_data, dataset2.train_data, dataset1.val_data,
                      dataset2.val_data):
            for pair in _data:
                pair['clean'] = preprocess_noisy(pair['clean'])
        # handle info leak
        all_val = dataset1.val_data + dataset2.val_data
        all_val = set([pair['clean'] for pair in all_val])

        all_data = (dataset1.train_data + dataset2.train_data + dataset1.val_data +
                    dataset2.val_data)
        train_data, valid_data = [], []
        for pair in all_data:
            if pair['clean'] in all_val:
                valid_data.append(pair)
            else:
                train_data.append(pair)

        save_dataset = TrainValDataset(train_data, valid_data)
        nsml.bind(save=save_dataset.save, load=save_dataset.load)
        nsml.save('corpus_data')

    elif args.mode == 'pretrain_corpus':
        dataset = TrainValDataset(None, None)
        nsml.bind(save=dataset.save, load=dataset.load)
        nsml.load(checkpoint='corpus_data',
                  session='KR95368/airush2021-2-4/' + args.data_session)
        if args.noise_type != 'all':
            dataset.train_data = [
                sent for sent in dataset.train_data if sent['type'] == args.noise_type
            ]
            dataset.val_data = [
                sent for sent in dataset.val_data if sent['type'] == args.noise_type
            ]
        train_corpus, valid_corpus = dataset.train_data, dataset.val_data
        train_data, train_label = read_strings(data_path), read_strings(label_path)
        train_data = [preprocess_noisy(sent) for sent in train_data]
        all_sents = [x['noisy'] for x in train_corpus
                    ] + [x['clean'] for x in train_corpus] + train_data + train_label

        if isinstance(tokenizer, CharTokenizer):
            tokenizer = CharTokenizer.from_strings(all_sents,
                                                   args.vocab_size,
                                                   max_seq_length=args.max_seq_length,
                                                   use_jamo=args.use_jamo)
            print('vocab size', len(tokenizer))
            print(tokenizer.vocab)

        bind_nsml(model, tokenizer, args)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, dim=1)

        train(model, tokenizer, train_corpus, valid_corpus, args, logger)

    elif args.mode == 'pretrain_pseudo_data':
        dataset = TrainValDataset(None, None)
        nsml.bind(save=dataset.save, load=dataset.load)
        nsml.load(checkpoint='pseudo_data',
                  session='KR95368/airush2021-2-4/' + args.data_session)
        pseudo_train_data, pseudo_valid_data = dataset.train_data, dataset.val_data
        if args.noise_type != 'all':
            pseudo_train_data = [
                sent for sent in pseudo_train_data if sent['type'] == args.noise_type
            ]
            pseudo_valid_data = [
                sent for sent in pseudo_valid_data if sent['type'] == args.noise_type
            ]
        logger.info(f'# of train corpus: {len(pseudo_train_data)}')
        logger.info(f'# of valid corpus: {len(pseudo_valid_data)}')

        train_data = read_strings(data_path)
        train_data = [preprocess_noisy(sent) for sent in train_data]
        all_sents = [x['noisy'] for x in pseudo_train_data
                    ] + [x['clean'] for x in pseudo_train_data] + train_data

        if isinstance(tokenizer, CharTokenizer):
            tokenizer = CharTokenizer.from_strings(all_sents,
                                                   args.vocab_size,
                                                   max_seq_length=args.max_seq_length,
                                                   use_jamo=args.use_jamo)
            print('vocab size', len(tokenizer))
            print(tokenizer.vocab)

        bind_nsml(model, tokenizer, args)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, dim=1)

        train(model, tokenizer, pseudo_train_data, pseudo_valid_data, args, logger)

    elif args.mode == 'eda':
        # do eda
        train_data, train_corpus = read_strings(data_path), read_strings(corpus_path)
        train_annotation = read_strings(annotation_path)
        train_label = read_strings(label_path)
        print('train data size:', len(train_data))
        for ex in random.sample(list(zip(train_data, train_label, train_annotation)), 50):
            print(ex)

        print('ends with period:',
              len([label for label in train_label if label.endswith('.')]))

        print('corpus_size', len(train_corpus)),
        for ex in random.sample(train_corpus, 50):
            print(ex)

    elif args.mode == 'train_spm':
        if args.spacing_cache_session:
            dataset = TrainValDataset(None, None)
            nsml.bind(save=dataset.save, load=dataset.load)
            nsml.load(checkpoint='spacing_cache',
                      session='KR95368/airush2021-2-4/' + args.spacing_cache_session)
            pairs = dataset.train_data + dataset.val_data
        else:
            pairs, _ = get_train_val_data(data_path,
                                          label_path,
                                          val_ratio=0,
                                          preprocess=bool(args.preprocess))
        train_sents = [x['noisy'] for x in pairs] + [x['clean'] for x in pairs]
        logger.info(f'# of sentences: {len(train_sents)}')

        tokenizer = SentencePienceTrainer(args)
        nsml.bind(save=tokenizer.save)
        if args.use_mecab:
            mecab_result = tokenizer.mecab_tokenize(train_sents)
            tokenizer.word_piece_train(iterator=mecab_result)
        else:
            tokenizer.word_piece_train(iterator=train_sents)
        nsml.save('tokenizer')

    elif args.mode == 'cache_spacing':
        train_data, valid_data = get_train_val_data(data_path,
                                                    label_path,
                                                    annotation_path=annotation_path,
                                                    val_ratio=args.val_ratio,
                                                    add_spacing=True)
        dataset = TrainValDataset(train_data, valid_data)
        nsml.bind(save=dataset.save, load=dataset.load)
        nsml.save('spacing_cache')


if __name__ == '__main__':
    main()
