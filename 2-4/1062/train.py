import os

from g2pk import G2p

import argparse
import json
import logging
import math

import random
import time

import numpy as np
import torch


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR

import pandas as pd

from model import TransformerModel
from model_AR import TransformerModel as TransformerModel_AR
from model_AR import GPTPretrain

from tokenizer import CharTokenizer
from dataset import TextDataset, collate_fn, make_smart_batches, make_smart_batches_pretrain
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu

import nsml
from nsml import DATASET_PATH

logging.basicConfig(format='%(asctime)s -  %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', 
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f'cuda? {torch.cuda.is_available()}')


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    # parser.add_argument("--num_val_data", type=int, default=1000)
    parser.add_argument("--num_val_data", type=int, default=896)
    # parser.add_argument("--num_val_data", type=int, default=768)

    # model
    parser.add_argument("--vocab_size", type=int, default=2366+4)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=4096)

    # training
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=5e-4)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=4000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=217) # 한 epoch당 2번씩 validation

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)

    # nsml
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default="0")

    args = parser.parse_args()
    return args


def set_seed(args):
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def calc_loss(model, batch):
    src, tgt, src_mask, tgt_mask, tgt_label, anno_labels = batch
    memory_conv, output = model(src=src, tgt=tgt, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)
    bsz = tgt.size(1)

    anno_dict = {'perfect':0, 'spacing':1, 'pasting':2, 'tense':3, 
                 'honorific':4, 'punctuation':5, 'typo':6, 'advanced':7}
    anno_dict_reversed = {v:k for k,v in anno_dict.items()}

    _, top7 = torch.topk(memory_conv, 7, dim=1)    
    # print(top7.shape) # torch.Size([128, 7])
    
    # check the answer
    anno_answer_dict = []
    for a, p in zip(anno_labels, top7) :
        answer = []
        for e, i in enumerate(a) :
            if i == 1 :
                answer.append(e)
        ans = sorted(pd.Series(answer).replace(anno_dict_reversed).tolist())
        predd = sorted(pd.Series(p.cpu().numpy()).replace(anno_dict_reversed).tolist()[:len(ans)])
        anno_answer_dict.append({'answer':ans, 'predict':predd})
    
    # annoation loss
    multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduction='none')
    multi_loss = multi_criterion(memory_conv, anno_labels).mean()

    raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')
    raw_loss = raw_loss.view(-1, bsz)
    loss = (raw_loss * tgt_mask.float()).sum(0).mean() + multi_loss
    items = [loss.data.item(), bsz, tgt_mask.sum().item()]

    return loss, items, multi_loss, anno_answer_dict


def evaluate(model, data_loader, args):
    model.eval()
    meter = Meter()
    anno_answer_dict_ = []
    with torch.no_grad():
        for batch in zip(data_loader[0], data_loader[1], data_loader[2], data_loader[3], data_loader[4], data_loader[5]):
            batch = tuple(t.to(args.device) for t in batch)
            _, items, anno_loss, anno_answer_dict = calc_loss(model, batch)
            anno_answer_dict_ += anno_answer_dict
            meter.add(*items)
    return meter.average(), meter.print_str(False), anno_loss, anno_answer_dict_


def correct(model, model2, tokenizer, test_data, args):
    model.eval()
    model2.eval()
    
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        batch = test_data[i:i + args.eval_batch_size]

        src_token_ids = [tokenizer(x) for x in batch]
        src_seq_length = [len(x) for x in src_token_ids]
        src_max_seq_length = max(src_seq_length)
        src_padded = []
        src_padding_mask = []
        for x in src_token_ids:
            x = x[:src_max_seq_length]
            src_pad_length = src_max_seq_length - len(x)
            src_padded.append(x + [1] * src_pad_length)
            src_padding_mask.append([1] * len(x) + [0] * src_pad_length)
        src_padded = torch.tensor(src_padded).t().contiguous().to(args.device)
        src_padding_mask = torch.tensor(src_padding_mask).bool().t().to(args.device)

        memory = model(src=src_padded, src_key_padding_mask=~src_padding_mask)
        memory2 = model2(src=src_padded, src_key_padding_mask=~src_padding_mask)

        tgt_token_ids = [[2] for _ in batch]
        end = [False for _ in batch]
        for l in range(src_max_seq_length + 7) :
            tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)

            output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)
            output2 = model2(tgt=tgt, memory=memory2, memory_key_padding_mask=~src_padding_mask)
            # output = model(tgt=tgt, memory=memory, memory_key_padding_mask=None)
            # output2 = model2(tgt=tgt, memory=memory2, memory_key_padding_mask=None)

            # output = output + output2

            # output = output*0.3 + output2*0.7
            # output = output*0.3 + output2*0.7 / length+7    # 82.96679608093679

            # output = output*0.35 + output2*0.65
            # output = output*0.35 + output2*0.65 / length+7    # 83.06344589069559

            # output = output*0.4 + output2*0.6
            # output = output*0.4 + output2*0.6 / length+7    # 83.12797479385937 **

            output = output*0.425 + output2*0.575
            # output = output*0.425 + output2*0.575 / length+7    # !

            # output = output*0.45 + output2*0.55
            # output = output*0.45 + output2*0.55 / length+7    # 83.12665650601816

            # output = output*0.5 + output2*0.5
            # output = output*0.5 + output2*0.5 / length+7   # 83.1235319490353
            
            # output = output*0.515 + output2*0.485
            # output = output*0.515 + output2*0.485 / length+7

            # output = output*0.525 + output2*0.475
            # output = output*0.525 + output2*0.475 / length+6    # 83.11656812008897
            # output = output*0.525 + output2*0.475 / length+7    # 83.1196447487809
            # output = output*0.525 + output2*0.475 / length+8    # 83.11572750022343
            # output = output*0.525 + output2*0.475 / length+9    #
            # output = output*0.525 + output2*0.475 / length+10   # 83.09945275854366
            # output = output*0.525 + output2*0.475 / length+12    # 

            # output = output*0.55 + output2*0.45
            # output = output*0.55 + output2*0.45 / length+10    # 83.0935123242386 
            # output = output*0.55 + output2*0.45 / length+15    # 83.04809312811813

            # output = output*0.575 + output2*0.425
            # output = output*0.575 + output2*0.425 / length+8
            # output = output*0.575 + output2*0.425 / length+10  # 83.09362882618873
            # output = output*0.575 + output2*0.425 / length+12

            # output = output*0.6 + output2*0.4
            # output = output*0.6 + output2*0.4 / length+5      # 83.06585259485391
            # output = output*0.6 + output2*0.4 / length+10     # 83.07103015844297
            # output = output*0.6 + output2*0.4 / length+20     # 82.9889948544473

            # output = output*0.65 + output2*0.35
            # output = output*0.65 + output2*0.35 / length+10   # 83.03664377594195

            # output = output*0.7 + output2*0.3

            top1 = output[-1].argmax(-1).tolist()
            for i, tok in enumerate(top1):
                if tok == 3 or l >= src_seq_length[i] + 7 :
                    end[i] = True
                tgt_token_ids[i].append(tok if not end[i] else 3)
            if all(end):
                break

        prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 4]) for x in tgt_token_ids])

    return prediction


def train(model, model2, tokenizer, train_data, corrupted_train, valid_data, args):

    model.train()
    model2.train()
    
    anno_dict = {'perfect':0, 'spacing':1, 'pasting':2, 'tense':3, 
                 'honorific':4, 'punctuation':5, 'typo':6, 'advanced':7}
    
    g2p = G2p()
    corrupted_train_data = []; corrupted_annotation = []
    for raw, cor in zip(train_data, corrupted_train) :
        raw_split = raw['clean'].split(' ')
        cor_split = cor['corrupt'].split(' ')

        if cor['idx_replace'] :
            
            # typo
            make_label = [0 for _ in range(8)]
            make_label[6] = 1
            num = max(len(cor['idx_replace'])//4, 1)
            idx_selected = random.sample(cor['idx_replace'], num)
            for random_idx in idx_selected :
                raw_split[random_idx] = cor_split[random_idx]
            noisy = ' '.join(raw_split)
            noisy_raw = str(noisy)
            
            # pasting
            num_space_error = random.choice([0,0,0,1,2,3])
            len_noisy = len(noisy)
            for _ in range(num_space_error) :
                random_place = random.choice(range(len_noisy))
                noisy = noisy[:random_place] +' '+ noisy[random_place:]
                noisy = noisy.replace('  ', ' ')
            noisy = noisy.strip()
            if (num_space_error > 0) and (noisy != noisy_raw) :
                make_label[2] = 1
            noisy_raw2 = str(noisy)
            
            # spacing
            noisy_split = noisy.split(' ')
            len_noisy_split = len(noisy_split)
            num_space_error = min(random.choice([0,0,0,1,2,3]), len_noisy_split)
            if len_noisy_split > 2 :
                for _ in range(num_space_error) :
                    random_place = random.choice(range(len_noisy_split-1))
                    pasting = str(noisy_split[random_place]) + str(noisy_split[random_place+1])
                    noisy_split[random_place] = pasting
                    del noisy_split[random_place+1]
                    len_noisy_split = len(noisy_split)
                    if len_noisy_split == 1 :
                        break
                noisy = ' '.join(noisy_split)
                noisy = noisy.strip()
                if (num_space_error > 0) and (noisy != noisy_raw2) :
                    make_label[1] = 1
                if noisy == noisy_raw :
                    make_label[2] = 0
            
            corrupted_train_data.append({'clean':raw['clean'], 'noisy':noisy, 'annotation':make_label})

    for row in corrupted_train_data[:30] :
        logger.info(row['clean'])
        logger.info(row['noisy'])
        logger.info(row['annotation'])
        logger.info(' ')

    logger.info(f'total train data: {len(train_data+corrupted_train_data)}')
    
    # train_dataloader, _ = make_smart_batches(train_data+corrupted_train_data, tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)
    train_dataloader, _ = make_smart_batches(train_data, tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)
    # train_dataloader, _ = make_smart_batches(train_data+random.sample(corrupted_train_data, 25000), tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)
    valid_dataloader, ordered_index = make_smart_batches(valid_data, tokenizer, args.eval_batch_size, anno_label=True, padding_token_id=1)
    
    valid_data = pd.Series(valid_data)[ordered_index].tolist()    
    valid_noisy = [x['noisy'] for x in valid_data]
    valid_clean = [x['clean'] for x in valid_data]
    
    
    epochs = (args.max_steps - 1) // len(train_dataloader[0]) + 1
    logger.info(f'total epochs: {epochs}')
    
    # model1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=eval(args.adam_betas), eps=args.eps,
                                 weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
    #                                 eps=args.eps,
    #                                 weight_decay=args.weight_decay)
    lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    # model2
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr,
                                  betas=eval(args.adam_betas), eps=args.eps,
                                  weight_decay=args.weight_decay)
    # optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=args.lr,
    #                                  eps=args.eps,
    #                                  weight_decay=args.weight_decay)
    lr_lambda2 = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler2 = LambdaLR(optimizer2, lr_lambda2)


    # train
    step = 0
    best_val_gleu = -float("inf")
    meter = Meter(); meter2 = Meter()
    for epoch in range(1, epochs + 1):
        
        # make dataset
        corrupted_train_data = []; corrupted_annotation = []
        for raw, cor in zip(train_data, corrupted_train) :
            raw_split = raw['clean'].split(' ')
            cor_split = cor['corrupt'].split(' ')
            
            if cor['idx_replace'] :
                
                # typo
                make_label = [0 for _ in range(8)]
                make_label[6] = 1
                num = max(len(cor['idx_replace'])//4, 1)
                idx_selected = random.sample(cor['idx_replace'], num)
                for random_idx in idx_selected :
                    raw_split[random_idx] = cor_split[random_idx]
                noisy = ' '.join(raw_split)
                noisy_raw = str(noisy)
                
                # pasting
                num_space_error = random.choice([0,0,0,1,2,3])
                len_noisy = len(noisy)
                for _ in range(num_space_error) :
                    random_place = random.choice(range(len_noisy))
                    noisy = noisy[:random_place] +' '+ noisy[random_place:]
                    noisy = noisy.replace('  ', ' ')
                noisy = noisy.strip()
                if (num_space_error > 0) and (noisy != noisy_raw) :
                    make_label[2] = 1
                noisy_raw2 = str(noisy)
                
                # spacing
                noisy_split = noisy.split(' ')
                len_noisy_split = len(noisy_split)
                num_space_error = min(random.choice([0,0,0,1,2,3]), len_noisy_split)
                if len_noisy_split > 2 :
                    for _ in range(num_space_error) :
                        random_place = random.choice(range(len_noisy_split-1))
                        pasting = str(noisy_split[random_place]) + str(noisy_split[random_place+1])
                        noisy_split[random_place] = pasting
                        del noisy_split[random_place+1]
                        len_noisy_split = len(noisy_split)
                        if len_noisy_split == 1 :
                            break
                    noisy = ' '.join(noisy_split)
                    noisy = noisy.strip()
                    if (num_space_error > 0) and (noisy != noisy_raw2) :
                        make_label[1] = 1
                    if noisy == noisy_raw :
                        make_label[2] = 0
                
                corrupted_train_data.append({'clean':raw['clean'], 'noisy':noisy, 'annotation':make_label})
        
        if epoch % 2 == 0 :
            train_dataloader, _ = make_smart_batches(train_data, tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)
        else :
            train_dataloader, _ = make_smart_batches(corrupted_train_data, tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)

        # train_dataloader, _ = make_smart_batches(train_data+corrupted_train_data, tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)
        # train_dataloader, _ = make_smart_batches(train_data+random.sample(corrupted_train_data, 25000), tokenizer, args.train_batch_size, anno_label=True, padding_token_id=1)


        for batch in zip(train_dataloader[0], train_dataloader[1], train_dataloader[2], train_dataloader[3], train_dataloader[4], train_dataloader[5]):
            step += 1
            batch = tuple(t.to(args.device) for t in batch)
            loss, items, multi_loss, _ = calc_loss(model, batch)
            loss2, items2, multi_loss2, _ = calc_loss(model2, batch)
            meter.add(*items)
            meter2.add(*items2)

            loss.backward(); loss2.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(model2.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer2.step()
            model.zero_grad(); model2.zero_grad()
            scheduler.step(); scheduler2.step()
            
            batch = tuple(t.to('cpu') for t in batch)

            if step % args.log_interval == 0:
                lr = scheduler.get_lr()[0]
                lr2 = scheduler2.get_lr()[0]
                loss_sent, loss_token = meter.average()
                loss_sent2, loss_token2 = meter2.average()

                logger.info(f' [{step:5d}] lr {lr:.6f} | {meter.print_str(True)} | annotation loss: {multi_loss:5.2f}')
                logger.info(f' [{step:5d}] lr {lr2:.6f} | {meter2.print_str(True)} | annotation loss: {multi_loss2:5.2f}')
                logger.info(' ')
                nsml.report(step=step, scope=locals(), summary=True,
                            train__lr=lr, train__loss_sent=loss_sent, train__token_ppl=math.exp(loss_token))
                meter.init()

            if step % args.eval_interval == 0:
                 # valiation and save per epoch(or step)
                start_eval = time.time()
                (val_loss, val_loss_token), valid_str, anno_loss, anno_answer_dict = evaluate(model, valid_dataloader, args)
                (val_loss2, val_loss_token2), valid_str2, anno_loss2, anno_answer_dict2 = evaluate(model2, valid_dataloader, args)

                prediction = correct(model, model2, tokenizer, valid_noisy, args)

                val_em = em(prediction, valid_clean)
                val_gleu = gleu(prediction, valid_clean)
        
                logger.info(f' [{step:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f} | annotation loss: {anno_loss:5.2f}')
                start_num = min(5*epoch, 1000)
                for p, n, c, an in zip(prediction[start_num:start_num+10], valid_noisy[start_num:start_num+10], 
                                       valid_clean[start_num:start_num+10], anno_answer_dict[start_num:start_num+10]) :
                    logger.info(f'( input) {n}')
                    logger.info(f'(  pred) {p}')
                    logger.info(f'(answer) {c}')
                    logger.info(f'(anno   pred) {an["predict"]}')
                    logger.info(f'(anno answer) {an["answer"]}')
                    logger.info('-' * 89)
        
                nsml.report(step=step, scope=locals(), summary=True,
                            valid__loss_sent=val_loss, valid__token_ppl=math.exp(val_loss_token),
                            valid__em=val_em, valid__gleu=val_gleu)
        
                if val_gleu > best_val_gleu :
                    nsml.save(str(step))
                    logger.info(f'save model, epoch {epoch}')
        
                if val_gleu > best_val_gleu :
                    best_val_gleu = val_gleu
        
                meter.start += time.time() - start_eval



def bind_nsml(model, model2, tokenizer=None, args=None):
    def save(path, **kwargs):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        torch.save(model2.state_dict(), open(os.path.join(path, 'model_2.pt'), 'wb'))
        if tokenizer is not None:
            tokenizer.save(os.path.join(path, 'vocab.txt'))

    def load(path, **kwargs):
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        model2.load_state_dict(torch.load(open(os.path.join(path, 'model_2.pt'), 'rb'),
                                          map_location=lambda storage, loc: storage))
        if tokenizer is not None:
            tokenizer.load(os.path.join(path, 'vocab.txt'))

    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''
        pred = correct(model, model2, tokenizer, test_data, args)
        print('test_data:', test_data)
        print('predict:', pred)
        return pred

    import nsml
    nsml.bind(save, load, infer)


def main():

    args = get_args()
    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # set_seed(args)

    model = TransformerModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    ).to(args.device)
    logger.info(f"# of model parameters: {sum(p.numel() for p in model.parameters()) * 1e-6:.2f}M")

    model2 = TransformerModel_AR(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    ).to(args.device)
    logger.info(f"# of model2 parameters: {sum(p.numel() for p in model2.parameters()) * 1e-6:.2f}M")

    tokenizer = CharTokenizer([])

    bind_nsml(model, model2, tokenizer, args)
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":

        nsml.load(checkpoint=17143, session='KR95412/airush2021-2-4/1004')
        nsml.save(17143)
        exit()
        
        noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
        clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))
        train_annotation = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
        train_corpus = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
        
        # train_len_diff = []
        # for n, c in zip(noisy_sents, clean_sents) :
        #     train_len_diff.append(len(n)-len(c))
        # print(pd.Series(train_len_diff).describe())
        '''
        count    57130.000000
        mean        -1.000945
        std          4.007734
        min        -38.000000
        25%         -3.000000
        50%         -1.000000
        75%          0.000000
        max         93.000000
        '''

        anno_dict = {'perfect':0, 'spacing':1, 'pasting':2, 'tense':3, 
                     'honorific':4, 'punctuation':5, 'typo':6, 'advanced':7}

        anno_labels = []
        for a in train_annotation :
            splitted = a.split(',')
            labels = [0 for _ in range(8)]
            for s in splitted :
                labels[anno_dict[s]] = 1
            anno_labels.append(labels)

        pairs = [{"noisy": noisy, "clean": clean, "annotation":annotation} 
                 for noisy, clean, annotation in zip(noisy_sents, clean_sents, anno_labels)]

        # train_data, valid_data = list(pairs), pairs[-args.num_val_data:]
        train_data, valid_data = pairs[:-args.num_val_data], pairs[-args.num_val_data:]
        logger.info(f"# of train data: {len(train_data)}")
        logger.info(f"# of valid data: {len(valid_data)}")

        g2p = G2p()
        corrupted_train = []
        # for i, sent in enumerate(train_data[:1000]) : ## !!!
        for i, sent in enumerate(train_data) : ## !!!
            raw = sent['clean']

            # # group_vowels, descriptive
            # d = random.sample([True, False], 1)[0]
            # g = random.sample([True, False], 1)[0]
            # corrupt = g2p(sent['clean'], descriptive=d, group_vowels=g)

            corrupt = g2p(sent['clean'])

            raw_split = raw.split(' ')
            corrupt_split = corrupt.split(' ')
            idx_replace = [idx for idx, (r, c) in enumerate(zip(raw_split, corrupt_split)) if r != c]
            
            if i % 1000 == 0 :
                logger.info(f'data corrputing, {i}th')
                logger.info(raw)
                logger.info(corrupt)
                logger.info(idx_replace)
                logger.info(' ')

            corrupted_train.append({'corrupt':corrupt, 'idx_replace':idx_replace})

        train_sents = [x['noisy'] for x in train_data] + [x['corrupt'] for x in corrupted_train] + [x['clean'] for x in train_data]
        tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)


        # train Auto regressize LM
        #############################
        gpt_batch = 256
        gpt_lr = 0.0001
        gpt_epoch = 50 ## !!!
        gpt_freeze = True
        train_sample = 1024 ## !!!
        #############################

        gpt = GPTPretrain(vocab_size=args.vocab_size,
                          hidden_size=args.hidden_size,
                          num_attention_heads=args.num_attention_heads,
                          num_decoder_layers=args.num_decoder_layers,
                          intermediate_size=args.intermediate_size,
                          dropout=args.dropout)
        gpt.to(args.device)

        pretrain_dataloader = make_smart_batches_pretrain(train_corpus, tokenizer, batch_size=gpt_batch, padding_token_id=1)        
        # pretrain_dataloader = make_smart_batches_pretrain(train_corpus[:train_sample], tokenizer, batch_size=gpt_batch, padding_token_id=1)
        
        criterion_lm = torch.nn.CrossEntropyLoss()
        optimizer_lm = torch.optim.Adam(gpt.parameters(), lr=gpt_lr)
        
        gpt.train()
        optimizer_lm.zero_grad()
        for epoch in range(1, gpt_epoch+1):
            losses = 0; step_cnt = 0
            for batch in zip(pretrain_dataloader[0], pretrain_dataloader[2]):
                batch = tuple(t.to(args.device) for t in batch)
                gpt_logit = gpt(batch[0])
                loss_lm = criterion_lm(gpt_logit.contiguous().view(-1, gpt_logit.size(-1)), batch[1].contiguous().view(-1))
                loss_lm.backward()
                optimizer_lm.step()
                optimizer_lm.zero_grad()
                losses += loss_lm.item()
                if step_cnt % 200 == 0 :
                    logger.info(f'epoch {epoch}, step {step_cnt},\t {loss_lm.item()}')
                step_cnt += 1
                batch = tuple(t.to('cpu') for t in batch)
            logger.info(f'epoch {epoch}. {losses/len(pretrain_dataloader[0])}')
        del batch
        
        # set GPT embedding layer
        model2.set_gpt_emb(gpt.gpt.token_embeddings, freeze=gpt_freeze)
        gpt.to('cpu')
        gpt.gpt.token_embeddings.to(args.device)

        bind_nsml(model, model2, tokenizer, args)

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model, dim=1)

    # if args.mode == "train":
        train(model, model2, tokenizer, train_data, corrupted_train, valid_data, args)


if __name__ == "__main__":
    main()
