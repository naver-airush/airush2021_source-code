import math
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import get_dataloader
from pretrain_dataset import get_pretrain_dataloader
from utils.meter import Meter
from utils.loss import LabelSmoothingNLLLoss
from evaluation import em, gleu

import nsml


def calc_loss(model, batch, ignore_idx=1, criterion=None):
    if len(batch) == 4:
        multitask = False
        src, tgt, src_mask, tgt_mask = batch  # [bs, T]
        output = model(src=src,
                       tgt=tgt[:, :-1],
                       src_key_padding_mask=src_mask,
                       tgt_key_padding_mask=tgt_mask[:, :-1])  # [T, bs, V]
    else:
        multitask = True
        src, tgt, src_mask, tgt_mask, annot_label = batch  # [bs, T]
        output, annot_pred = model(src=src,
                                   tgt=tgt[:, :-1],
                                   src_key_padding_mask=src_mask,
                                   tgt_key_padding_mask=tgt_mask[:, :-1],
                                   do_multitask=True)  # [T, bs, V], [bs, 8]
    output = output.permute(1, 2, 0)  # [bs, V, T]
    target_label = tgt[:, 1:]  # [bs, T]
    bsz = tgt.shape[0]
    if criterion is None:
        criterion = F.nll_loss
    loss = criterion(output, target_label, ignore_index=ignore_idx)
    if multitask:
        loss += F.binary_cross_entropy_with_logits(annot_pred, annot_label)
    items = [loss.data.item(), bsz, (~tgt_mask).sum().item()]
    return loss, items


@torch.no_grad()
def evaluate(model, data_loader, args, criterion=None):
    model.eval()
    meter = Meter()
    for batch in data_loader:
        batch = tuple(t.to(args.device) for t in batch)
        _, items = calc_loss(model,
                             batch,
                             ignore_idx=data_loader.dataset.pad_idx,
                             criterion=criterion)
        meter.add(*items)
    return meter.average(), meter.print_str(False)


@torch.no_grad()
def correct(model, tokenizer, test_data, args, debug=False, logger=None):
    model.eval()
    prediction = []
    test_loader = get_dataloader(args,
                                 test_data,
                                 tokenizer,
                                 'test',
                                 drop=args.token_dropout)
    eos_idx = test_loader.dataset.eos_idx
    sos_idx = test_loader.dataset.sos_idx
    pad_idx = test_loader.dataset.pad_idx
    for i, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        src_padded, src_padding_mask = batch  # [bs, S]

        tgt_token_ids = model.generate(src_padded, src_padding_mask, sos_idx, eos_idx,
                                       pad_idx)

        prediction.extend(tokenizer.decode(tgt_token_ids, skip_special_tokens=True))
        if logger is not None and i % args.log_interval == 0:
            logger.info('step {}/{}'.format(i, len(test_loader)))
    if debug:
        print(prediction)
    return prediction


def train(model, tokenizer, train_data, valid_data, args, logger, unlabeled_data=None):
    model.train()

    dataloader_fn = get_dataloader if args.mode in (
        'train', 'finetune') else get_pretrain_dataloader

    train_dataloader = dataloader_fn(args,
                                     train_data,
                                     tokenizer,
                                     'train',
                                     do_multitask=bool(args.do_multitask),
                                     drop=args.token_dropout)
    if valid_data is not None:
        valid_dataloader = dataloader_fn(args,
                                         valid_data,
                                         tokenizer,
                                         'val',
                                         do_multitask=bool(args.do_multitask),
                                         drop=args.token_dropout)
        valid_noisy = [x['noisy'] for x in valid_data]
        valid_clean = [x['clean'] for x in valid_data]

    if unlabeled_data is not None:
        unlabeled_dataloader = dataloader_fn(args, unlabeled_data, tokenizer, 'unlabeled')
        unlabeled_iterator = iter(unlabeled_dataloader)

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     betas=eval(args.adam_betas),
                                     eps=args.eps,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(),
                          lr=args.lr,
                          betas=eval(args.adam_betas),
                          eps=args.eps,
                          weight_decay=args.weight_decay)

    if args.scheduler == 'lambda':
        lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (
            x / args.num_warmup_steps)**-0.5
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_steps)
    if args.label_smoothing > 0:
        criterion = LabelSmoothingNLLLoss(smoothing=args.label_smoothing)
    else:
        criterion = None

    step = 0
    best_val_gleu = -float('inf')
    best_val_loss = float('inf')
    meter = Meter()
    for _ in range(1, epochs + 1):
        for batch in train_dataloader:
            step += 1
            batch = tuple(t.to(args.device) for t in batch)
            loss, items = calc_loss(model,
                                    batch,
                                    train_dataloader.dataset.pad_idx,
                                    criterion=criterion)
            if unlabeled_data is not None:
                try:
                    unlabeled_batch = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_dataloader)
                    unlabeled_batch = next(unlabeled_iterator)
                unlabeled_batch = tuple(t.to(args.device) for t in unlabeled_batch)
                src_padded, src_padding_mask = unlabeled_batch  # [bs, S]

                tgt_token_ids = model.generate(src_padded, src_padding_mask,
                                               tokenizer.sos_idx, tokenizer.eos_idx,
                                               tokenizer.pad_idx)
                target_text = tokenizer.decode(tgt_token_ids, skip_special_tokens=True)
                tgt_token_ids, tgt_padding_mask = tokenizer(target_text, is_target=True)
                tgt_token_ids = tgt_token_ids.to(args.device)
                tgt_padding_mask = tgt_padding_mask.to(args.device)
                unlabeled_batch = (src_padded, tgt_token_ids, src_padding_mask,
                                   tgt_padding_mask)
                ssl_loss, ssl_items = calc_loss(model,
                                                unlabeled_batch,
                                                tokenizer.pad_idx,
                                                criterion=criterion)

                loss += ssl_loss
                items = [it + s_it for it, s_it in zip(items, ssl_items)]
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
                nsml.report(step=step,
                            scope=locals(),
                            summary=True,
                            train__lr=lr,
                            train__loss_sent=loss_sent,
                            train__token_ppl=math.exp(loss_token))
                meter.init()

            if step % args.eval_interval == 0:
                if valid_data is None:
                    logger.info('saving checkpoint for step {}'.format(step))
                    nsml.save(str(step))
                    if step >= args.max_steps:
                        break
                    if step >= args.turn_off_reg_step >= 0:
                        model.eval()
                    continue
                start_eval = time.time()
                (val_loss, val_loss_token), valid_str = evaluate(model,
                                                                 valid_dataloader,
                                                                 args,
                                                                 criterion=criterion)

                log_str = f' [{step:6d}] valid | {valid_str} '
                report_kwargs = {
                    'valid__loss_sent': val_loss,
                    'valid__token_ppl': math.exp(val_loss_token),
                }
                if args.mode in ('train', 'finetune'):
                    prediction = correct(model, tokenizer, valid_noisy, args)
                    print('examples')
                    for ex in list(zip(valid_noisy, prediction, valid_clean))[:10]:
                        print(ex)
                    val_em = em(prediction, valid_clean)
                    val_gleu = gleu(prediction, valid_clean)
                    log_str += f'| em {val_em:5.2f} | gleu {val_gleu:5.2f}'
                    report_kwargs['valid__em'] = val_em
                    report_kwargs['valid__gleu'] = val_gleu

                logger.info('-' * 89)
                logger.info(log_str)
                logger.info('-' * 89)
                nsml.report(step=step, scope=locals(), summary=True, **report_kwargs)

                if args.mode in ('train', 'finetune') and val_gleu > best_val_gleu:
                    best_val_gleu = val_gleu
                    logger.info('new best model found! Saving..')
                    nsml.save('best')
                elif 'pretrain' in args.mode and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info('new best model found! Saving..')
                    nsml.save('best')
                meter.start += time.time() - start_eval
                if args.turn_off_reg_step < 0 or step < args.turn_off_reg_step:
                    model.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
