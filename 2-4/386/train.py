import argparse
import json
import logging
import math
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

os.environ["HOME"] = "/home/nsml"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model import TransformerModel, EncoderDecoder, BertGpt2, TransformerS2Model
from tokenizer import CharTokenizer, CustomTokenizer
from dataset import TextDataset, collate_fn, collate_fn_pretrain, collate_fn_transformer, collate_fn_pretrain_transformer, collate_fn_mlm_pretrain_transformer
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu

import larva
from larva import LarvaTokenizer, LarvaModel, list_larva_models

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


def get_lr(optimizer : transformers.optimization) -> float:
    """Get learning_rate.
    
    Args:
        optimizer (transformers.optimization).
    
    Return:
        Learning_Rate (float).
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calc_loss(model, batch, args):
    src, tgt, src_mask, tgt_mask, tgt_label = batch
    if args.model == 'transformer':
        output = model(src=src, tgt=tgt, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)
    elif args.model in ['encoderdecoder', 'BertGpt2']:
        output = model(src=src, tgt=tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
    elif args.model == 'bertfuse':
        output = model(src_tokens=src, prev_output_tokens=tgt, bert_input=src)

    if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
        mask_tgt_label = tgt_label.clone()
        mask_tgt_label[tgt_label.eq(1)] = -100
    else:
        mask_tgt_label = tgt_label.clone()
        mask_tgt_label[tgt_label.eq(0)] = -100

    raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), mask_tgt_label.view(-1), reduction='none')
    # raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')

    if args.model == 'transformer':
        bsz = tgt.size(1)
        raw_loss = raw_loss.view(-1, bsz)
    else:
        bsz = tgt.size(0)
        raw_loss = raw_loss.view(bsz, -1)

    if args.model == 'transformer':
        loss = (raw_loss * tgt_mask.float()).sum(0).mean()
    else:
        loss = (raw_loss * tgt_mask.float()).sum(-1).mean()

    items = [loss.data.item(), bsz, tgt_mask.sum().item()]
    return loss, items


def calc_loss1(model, batch, args):
    src, tgt, src_mask, tgt_mask, tgt_label = batch

    print(f"source padded is {src}, source padded's shape is {src.shape}")
    print(f"target padded before -1 is {tgt}, target padded before -1 's shape is {tgt.shape}")
    print(f"source padding mask is {src_mask}, source padding mask's shape is {src_mask.shape}")
    print(f"target padding mask is {tgt_mask}, target padding mask's shape is {tgt_mask.shape}")
    print(f"target padded after 1 is {tgt_label}, target padded after 1 's shape is {tgt_label.shape}")

    if args.model == 'transformer':
        output = model.forward1(src=src, tgt=tgt, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)
    elif args.model in ['encoderdecoder', 'BertGpt2']:
        output = model.forward1(src=src, tgt=tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
    elif args.model == 'bertfuse':
        output = model(src_tokens=src, prev_output_tokens=tgt, bert_input=src)

    print(f"output shape is {output.shape}")

    if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
        mask_tgt_label = tgt_label.clone()
        mask_tgt_label[tgt_label.eq(1)] = -100
    else:
        mask_tgt_label = tgt_label.clone()
        mask_tgt_label[tgt_label.eq(0)] = -100

    print(f"cross entropy first input is {output.view(-1, output.size(-1)).shape}, second input is {tgt_label.view(-1).shape}")

    raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), mask_tgt_label.view(-1), reduction='none')
    # raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')

    if args.model == 'transformer':
        bsz = tgt.size(1)
        raw_loss = raw_loss.view(-1, bsz)
    else:
        bsz = tgt.size(0)
        raw_loss = raw_loss.view(bsz, -1)

    print(f"raw_loss is {raw_loss}, raw_loss 's shape is {raw_loss.shape}")
    if args.model == 'transformer':
        loss = (raw_loss * tgt_mask.float()).sum(0).mean()
    else:
        print(f"sum of raw_loss is {(raw_loss * tgt_mask.float()).sum(-1)}, sum of raw_loss's shape is {(raw_loss * tgt_mask.float()).sum(-1).shape}")
        loss = (raw_loss * tgt_mask.float()).sum(-1).mean()

    print(f"loss is {loss}, loss 's shape is {loss.shape}")

    items = [loss.data.item(), bsz, tgt_mask.sum().item()]
    return loss, items


def evaluate(model, data_loader, args):
    model.eval()
    meter = Meter()
    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(args.device) for t in batch)
            _, items = calc_loss(model, batch, args)
            meter.add(*items)
    return meter.average(), meter.print_str(False)


def correct(model, tokenizer, test_data, args):
    model.eval()
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        batch = test_data[i:i + args.eval_batch_size]

        if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            # src_token_ids = [tokenizer(x) for x in batch]
            src_token_ids = [[2] + tokenizer(x) + [3] for x in batch]
        elif args.tokenizer in ['dsksd', 'plus', 'wordpiece']:
            src_token_ids = tokenizer(batch).input_ids

        if args.model == 'transformer':
            src_seq_length = [len(x) for x in src_token_ids]
            src_max_seq_length = max(src_seq_length)
            src_padded = []
            src_padding_mask = []
            for x in src_token_ids:
                x = x[:src_max_seq_length]
                src_pad_length = src_max_seq_length - len(x)
                src_padded.append(x + [tokenizer.pad_token_id] * src_pad_length)
                src_padding_mask.append([1] * len(x) + [0] * src_pad_length)
            
            src_padded = torch.tensor(src_padded).t().contiguous().to(args.device)
            src_padding_mask = torch.tensor(src_padding_mask).bool().t().to(args.device)

            memory = model(src=src_padded, src_key_padding_mask=~src_padding_mask)

            tgt_token_ids = [[2] for _ in batch]
            end = [False for _ in batch]
            for l in range(src_max_seq_length + 20):
                tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)
                output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)
                top1 = output[-1].argmax(-1).tolist()
                for i, tok in enumerate(top1):
                    if tok == 3 or l >= src_seq_length[i] + 20:
                        end[i] = True
                    tgt_token_ids[i].append(tok if not end[i] else 3)
                if all(end):
                    break
        elif args.model in ['encoderdecoder', 'BertGpt2']:
            src_seq_length = [len(x) for x in src_token_ids]
            src_max_seq_length = max(src_seq_length)
    
            src_padded = []
            for x in src_token_ids:
                x = x[:src_max_seq_length]
                src_pad_length = src_max_seq_length - len(x)
                src_padded.append(x + [tokenizer.pad_token_id] * src_pad_length)
            
            src_padded = torch.LongTensor(src_padded).contiguous().to(args.device)
            src_padding_mask = src_padded.ne(tokenizer.pad_token_id).to(args.device)

            memory = model.encoder(
                input_ids=src_padded,
                attention_mask=src_padding_mask,
            )

            tgt_token_ids = [[2] for _ in batch]
            end = [False for _ in batch]
            for l in range(src_max_seq_length + 20):
                tgt_padded = torch.LongTensor(tgt_token_ids).contiguous().to(args.device)
                tgt_padding_mask = tgt_padded.ne(tokenizer.pad_token_id).to(args.device)

                # output = model(src=src_padded, tgt=tgt_padded, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
                output = model(tgt=tgt_padded, memory=memory, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
                top1 = output[:,-1,:].argmax(-1).tolist()

                for i, tok in enumerate(top1):
                    if tok == 3 or l >= src_seq_length[i] + 20:
                        end[i] = True
                    tgt_token_ids[i].append(tok if not end[i] else 3)
                if all(end):
                    break
        elif args.model == 'bertfuse':
            src_seq_length = [len(x) for x in src_token_ids]
            src_max_seq_length = max(src_seq_length)
    
            src_padded = []
            for x in src_token_ids:
                x = x[:src_max_seq_length]
                src_pad_length = src_max_seq_length - len(x)
                src_padded.append(x + [tokenizer.pad_token_id] * src_pad_length)
            
            src_padded = torch.LongTensor(src_padded).contiguous().to(args.device)
            bert_encoder_padding_mask = src_padded.eq(tokenizer.pad_token_id)
            
            bert_encoder_out =  model.bert_encoder(src_padded, attention_mask=~bert_encoder_padding_mask)
            bert_encoder_out = bert_encoder_out.last_hidden_state
            
            bert_encoder_out = bert_encoder_out.permute(1,0,2).contiguous()
            bert_encoder_out = {
                'bert_encoder_out': bert_encoder_out,
                'bert_encoder_padding_mask': bert_encoder_padding_mask,
            }
        
            encoder_out = model.encoder(src_padded, bert_encoder_out=bert_encoder_out)

            tgt_token_ids = [[2] for _ in batch]
            end = [False for _ in batch]
            for l in range(src_max_seq_length + 20):
                tgt_padded = torch.LongTensor(tgt_token_ids).contiguous().to(args.device)

                output = model.decoder(tgt_padded, encoder_out=encoder_out, bert_encoder_out=bert_encoder_out)[0]

                top1 = output[:,-1,:].argmax(-1).tolist()

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


def mlm_pretrain(args : argparse.Namespace, model : nn.Module, loader : DataLoader, optimizer : transformers.optimization, scheduler,
                tokenizer : BertTokenizer, loss_fnc_pretrain : nn.CrossEntropyLoss, epoch : int) -> None:
    """Masked Language Model Pretrain
    Args:
        args (argparse.Namespace)
        model (nn.Module)
        loader (DataLoader)
        optimizer (transformers.optimization)
        tokenizer (BertTokenizer)
        loss_fnc_pretrain (nn.CrossEntropyLoss): mlm pretrain loss
        epoch (int)
    """
    model.train()
    for step, batch in enumerate(loader):
        src, tgt, src_mask, tgt_mask, tgt_label = tuple(t.to(args.device) for t in batch)

        logits, labels = model.forward_pretrain(src, src_mask, tokenizer)
        loss = loss_fnc_pretrain(logits.view(-1, args.vocab_size), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % 100 == 0:
            current_lr = get_lr(optimizer)
            epochs = args.num_pretrain_epochs
            print(
                f"[{epoch + 1}/{epochs}] [{step}/{len(loader)}] loss : {loss.item()}  lr : {current_lr}"
            )


def train(model, tokenizer, train_data, valid_data, corpuses, args, index = None):
    global best_val_gleu

    train_dataset = TextDataset(train_data)

    print(f"\nfirst train_dataset is {train_dataset[0]}")

    if args.model == 'transformer':
        if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            collate_func = collate_fn
        else:
            collate_func = collate_fn_pretrain
    elif args.model in ['encoderdecoder', 'BertGpt2', 'bertfuse']:
        if args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            collate_func = collate_fn_transformer    
        elif args.tokenizer in ['plus', 'wordpiece']:
            collate_func = collate_fn_pretrain_transformer

    train_dataloader = DataLoader(
        train_dataset, 
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate_func(x, tokenizer, args.max_seq_length),
    )

    if valid_data is not None:
        valid_dataset = TextDataset(valid_data)
        valid_dataloader = DataLoader(
            valid_dataset, 
            sampler=SequentialSampler(valid_dataset),
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            collate_fn=lambda x: collate_func(x, tokenizer, args.max_seq_length),
        )

        valid_noisy = [x['noisy'] for x in valid_data]
        valid_clean = [x['clean'] for x in valid_data]

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), 
        lr=args.lr,
        betas=eval(args.adam_betas), 
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    if args.scheduler == 'lambda':
        lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'linear':
        warmup_steps = args.max_steps * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps
        )

    if args.mlm_pre and (index is None or index == 0):
        print('mlm_pretrain!!!')
        pre_optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(), 
            lr=args.lr,
            betas=eval(args.adam_betas), 
            eps=args.eps,
            weight_decay=args.weight_decay
        )
        loss_fnc_pretrain = nn.CrossEntropyLoss()

        if args.tokenizer == 'char':
            t_total = len(train_dataloader) * args.num_pretrain_epochs
            warmup_steps = t_total * args.warmup_ratio
            pre_scheduler = get_linear_schedule_with_warmup(
                pre_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            for epoch in range(args.num_pretrain_epochs):
                mlm_pretrain(args, model, train_dataloader, pre_optimizer, pre_scheduler, tokenizer, loss_fnc_pretrain, epoch)
        elif args.tokenizer == 'pre_char':
            corpus_dataset = TextDataset(corpuses)
            corpus_dataloader = DataLoader(
                corpus_dataset, 
                sampler=RandomSampler(corpus_dataset),
                batch_size=args.train_batch_size,
                num_workers=args.num_workers,
                collate_fn=lambda x: collate_fn_pretrain(x, tokenizer, args.max_seq_length),
            )
            t_total = len(corpus_dataloader) * args.num_pretrain_epochs
            warmup_steps = t_total * args.warmup_ratio
            pre_scheduler = get_linear_schedule_with_warmup(
                pre_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            for epoch in range(args.num_pretrain_epochs):
                mlm_pretrain(args, model, corpus_dataloader, pre_optimizer, pre_scheduler, tokenizer, loss_fnc_pretrain, epoch)
        elif args.tokenizer == 'wordpiece':
            corpus_dataset = TextDataset(corpuses)
            corpus_dataloader = DataLoader(
                corpus_dataset, 
                sampler=RandomSampler(corpus_dataset),
                batch_size=args.train_batch_size,
                num_workers=args.num_workers,
                collate_fn=lambda x: collate_fn_mlm_pretrain_transformer(x, tokenizer, args.max_seq_length),
            )
            t_total = len(corpus_dataloader) * args.num_pretrain_epochs
            warmup_steps = t_total * args.warmup_ratio
            pre_scheduler = get_linear_schedule_with_warmup(
                pre_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            for epoch in range(args.num_pretrain_epochs):
                mlm_pretrain(args, model, corpus_dataloader, pre_optimizer, pre_scheduler, tokenizer, loss_fnc_pretrain, epoch)



    step = 0
    if index is None or index == 0:
        best_val_gleu = -float("inf")
    meter = Meter()

    for epoch in range(1, epochs + 1):
        for batch in train_dataloader:
            model.train()
            step += 1
            batch = tuple(t.to(args.device) for t in batch)

            if step < 4:
                loss, items = calc_loss1(model, batch, args)
            else:
                loss, items = calc_loss(model, batch, args)

            if step < 4:
                print(f"first item loss is {items[0]}")
                print(f"second item bsz is {items[1]}")
                print(f"third item target mask sum is {items[2]}")
                print()

            meter.add(*items)

            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            scheduler.step()

            if step % args.log_interval == 0:
                lr = scheduler.get_lr()[0]
                loss_sent, loss_token = meter.average()

                logger.info(f' [{step:5d}] lr {lr:.6f} | {meter.print_str(True)}')
                nsml.report(
                    step=step,
                    scope=locals(),
                    summary=True,
                    train__lr=lr,
                    train__loss_sent=loss_sent,
                    train__token_ppl=math.exp(loss_token)
                )
                meter.init()

            if step % args.eval_interval == 0 and valid_data is not None:
                start_eval = time.time()
                (val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)
                prediction = correct(model, tokenizer, valid_noisy, args)

                for (i, (pred, clean)) in enumerate(zip(prediction, valid_clean)):
                    if i == 5:
                        break
                    print(f"{i + 1}'s pred sentence is {pred}, clean sentence is {clean}")

                val_em = em(prediction, valid_clean)
                val_gleu = gleu(prediction, valid_clean)

                logger.info('-' * 89)
                logger.info(f' [{step:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f}')
                logger.info('-' * 89)
                nsml.report(
                    step=step, 
                    scope=locals(), 
                    summary=True,
                    valid__loss_sent=val_loss, 
                    valid__token_ppl=math.exp(val_loss_token),
                    valid__em=val_em, 
                    valid__gleu=val_gleu
                )

                if val_gleu > best_val_gleu:
                    if args.kfold:
                        if index == args.n_splits - 1:
                            print('------------save model------------')
                            nsml.save(f"model_{epoch}")
                    else:
                        print('------------save model------------')
                        best_val_gleu = val_gleu
                        nsml.save("best")
                meter.start += time.time() - start_eval

            elif step % args.eval_interval == 0 and valid_data is None:
                if epoch >= 5:
                    print('------------save model------------')
                    nsml.save(f"model_{epoch}")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break


def bind_nsml(model, tokenizer=None, args=None):
    def save(path, **kwargs):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if tokenizer is not None and args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            tokenizer.save(os.path.join(path, 'vocab.txt'))
        elif args.tokenizer == 'wordpiece':
            tokenizer.save(os.path.join(path, 'wordpiece_tokenizer'))


    def load(path, **kwargs):
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        if tokenizer is not None and args.tokenizer in ['char', 'pre_char', 'kfold_char']:
            tokenizer.load(os.path.join(path, 'vocab.txt'))
        elif args.tokenizer == 'wordpiece':
            tokenizer.load(os.path.join(path, 'wordpiece_tokenizer'))


    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''
        return correct(model, tokenizer, test_data, args)

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
    elif args.tokenizer == 'dsksd':
        tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal')
        print(f"vocab size is {len(tokenizer)}")
        args.vocab_size = len(tokenizer)
    elif args.tokenizer == 'plus':
        tokenizer = LarvaTokenizer.from_pretrained("larva-kor-plus-base-cased")
        print(f"vocab size is {len(tokenizer)}")
        args.vocab_size = len(tokenizer)
    elif args.tokenizer == 'wordpiece':
        tokenizer = CustomTokenizer(args)



    if args.model == 'transformer':
        model = TransformerModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model == 'encoderdecoder':
        model = EncoderDecoder(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model == 'BertGpt2':
        model = BertGpt2(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model == 'bertfuse':
        model = TransformerS2Model(tokenizer, args=args).to(args.device)

    logger.info(f"# of model parameters: {sum(p.numel() for p in model.parameters()) * 1e-6:.2f}M")

    bind_nsml(model, tokenizer, args)
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":
        noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
        annotations = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
        corpuses = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
        clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

        print(f"noisy_sents length is {len(noisy_sents)}")
        print(f"annotations length is {len(annotations)}")
        print(f"clean_sents length is {len(clean_sents)}")
        print(f"corpuses length is {len(corpuses)}")

        for (i, noisy_sent) in enumerate(noisy_sents[:5]):
            print(f"{i + 1}'s noisy sentence is {noisy_sent}")

        for (i, clean_sent) in enumerate(clean_sents[:5]):
            print(f"{i + 1}'s clean sentence is {clean_sent}")

        for (i, annotation) in enumerate(annotations[:5]):
            print(f"{i + 1}'s annotation is {annotation}")

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
        elif args.tokenizer == 'pre_char':
            tokenizer = CharTokenizer.from_strings(corpuses + clean_sents + noisy_sents, args.vocab_size)
        elif args.tokenizer == 'kfold_char':
            tokenizer = CharTokenizer.from_strings(clean_sents + noisy_sents, args.vocab_size)
        elif args.tokenizer == 'wordpiece':
            tokenizer.get_tokenizer()
            print(f"vocab size is {len(tokenizer)}")
            args.vocab_size = len(tokenizer)

        bind_nsml(model, tokenizer, args)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, dim=1)

    if args.mode == "train":
        if not args.kfold:
            train(model, tokenizer, train_data, valid_data, corpuses, args)
        else:
            kfold = KFold(n_splits=args.n_splits)
            for index, (train_idx, valid_idx) in enumerate(kfold.split(range(len(pairs)))):
                train_data, valid_data = [pairs[idx] for idx in train_idx], [pairs[idx] for idx in valid_idx]
                train(model, tokenizer, train_data, valid_data, corpuses, args, index)


if __name__ == "__main__":
    main()
