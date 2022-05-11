import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import nsml
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings(action='ignore')
from torch.utils.data import TensorDataset, DataLoader


def Trainer(model, train, valid):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    batch_size = 1024
    learning_rate = 2e-4
    model.to(device)
    input_x = train['prod_nm']
    # input_x2 = train['measurement']
    # input_x2 = torch.Tensor(train['measurement'])
    input_y = torch.LongTensor(train['label'])
    train_loader = DataLoader(input_x, batch_size=batch_size)
    # train_loader2 = DataLoader(input_x2, batch_size=batch_size)
    train_y_loader = DataLoader(input_y, batch_size=batch_size)
    valid_x = valid['prod_nm']
    # valid_x2 = torch.Tensor(valid['measurement'])
    # valid_x2 = valid['measurement']
    # valid_dataset = TensorDataset(valid_x, valid_x2)
    valid_y = torch.LongTensor(valid['label'])
    valid_loader = DataLoader(valid_x, batch_size=batch_size)
    # valid_loader2 = DataLoader(valid_x2, batch_size=batch_size)
    valid_y_loader = DataLoader(valid_y, batch_size=batch_size)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps = 1e-8)
    print(f'batch_size = {batch_size}')
    print(f'learning_rate = {learning_rate}')
    loss_fn = nn.CrossEntropyLoss()
    max_val = 0
    check_cnt = 1
    best_val_loss = 10
    num_epochs = 70
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for batch, batch_y in zip(train_loader, train_y_loader):
        # for batch, batch2, batch_y in zip(train_loader, train_loader2, train_y_loader):
            batch_input_x = [t for t in batch]
            # batch_input_x2 = [t for t in batch2]
            # batch_input_x, batch_input_x2 = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(batch_input_x)
            # logits = model(batch_input_x, batch_y.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()


        model.eval()

        val_loss = 0
        val_accuracy = []
        for batch, batch_y in zip(valid_loader, valid_y_loader):
        # for batch, batch2, batch_y in zip(valid_loader, valid_loader2, valid_y_loader):
            batch_input_x = [t for t in batch]
            # batch_input_x2 = [t for t in batch2]
            # batch_input_x, batch_input_x2 = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(batch_input_x)
                # logits = model(batch_input_x, batch_y.to(device))

            loss = loss_fn(logits, batch_y.to(device))
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == batch_y.to(device)).cpu().numpy().mean()
            val_accuracy.append(accuracy)
        # if np.mean(val_accuracy) > max_val:
        if  best_val_loss > val_loss/len(valid_loader):
            max_val = np.mean(val_accuracy)
            best_model = model
            best_epoch = epoch + 1
            best_train_loss = train_loss/len(train_loader)
            best_val_loss = val_loss/len(valid_loader)
            # nsml.save(f'{epoch}')
            # cnt+=1
            check_cnt = 1
        else:
            check_cnt += 1
            if check_cnt == 5:
                break
        print('epoch: ', epoch + 1, 'train_loss: ', train_loss/len(train_loader), 'val_loss: ', val_loss/len(valid_loader), 'val_accuracy: ', np.mean(val_accuracy))
    print('best_model: ','epoch: ', best_epoch, 'train_loss: ', best_train_loss, 'val_loss: ', best_val_loss, 'val_accuracy: ', max_val)

    return best_model






