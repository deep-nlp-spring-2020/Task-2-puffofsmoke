#  Copyright (c) 2020.
#  roman.grigorov@gmail.com aka PuffOfSmoke or dePuff
import copy
import datetime
import traceback

import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import assignment.const as const


def train_eval_loop(model, criterion, dataset_train, dataset_val=None, optimizer=None, dataloader=DataLoader,
                    collate_fn=None, device=None,
                    shuffle_train=True,
                    epoch_n=10, batch_size=32, lr=1e-3, l2_reg_alpha=0,
                    early_stopping_patience=10, lr_scheduler=None, debug=True, writer=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    # else:
    #     optimizer = optimizer(model.parameters(), lr=lr)

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)

    dataloader_train = dataloader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
    dataloader_val = dataloader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            if debug:
                epoch_start2print = epoch_start.strftime('%d.%m.%Y %H:%M:%S')
                print('*' * 28)
                print(f'Epoch {epoch_i} started. {epoch_start2print}')

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(dataloader_train):

                # No fix because no more this bicycle to use
                # batch_x = batch.text.to(device)
                # batch_y = batch.label.to(device)

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y.flatten())

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += loss.item()
                train_batches_n += 1

                if writer:
                    writer.add_scalar('Loss/Train', loss.item(), epoch_i * len(dataloader_train) + batch_i)

            mean_train_loss /= train_batches_n

            duration = (datetime.datetime.now() - epoch_start).total_seconds()

            if debug:
                print(f'Iterations: {train_batches_n}')
                print(f'Duration: {duration:.2} sec')
                print(f'Average train loss: {mean_train_loss}')
                print('+' * 28)

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(dataloader_val):

                    # No fix because no more this bicycle to use
                    # batch_x = batch.text.to(device)
                    # batch_y = batch.label.to(device)

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    pred, _ = model(batch_x)
                    loss = criterion(pred, batch_y.flatten())

                    mean_val_loss += loss.item()
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            if debug:
                print(f'Average validation loss: {mean_val_loss}')
            if writer:
                writer.add_scalar('Loss/Val', mean_val_loss, (epoch_i + 1) * len(dataloader_train))

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                if debug:
                    print('>>> New Best Model <<<')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                if debug:
                    print(f'No model improvement in last {early_stopping_patience}. Let\'s stop it')
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)
            if debug:
                print('*' * 28)

        except KeyboardInterrupt:
            print('Stopped by User')
            break
        except Exception as ex:
            print('Got error: {}\n{}'.format(ex, traceback.format_exc()))
            break
    return best_val_loss, best_model


def model_performance_calculator(model, iterator):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0
    model.eval()
    for batch in iterator:
        x = batch.text
        y = batch.label

        y_pred, _ = model(x)

        predicted_classes = torch.round(torch.sigmoid(y_pred))
        target_classes = y

        tp += torch.sum((predicted_classes == target_classes) * predicted_classes == 1).float()
        fp += torch.sum((predicted_classes != target_classes) * predicted_classes == 1).float()
        fn += torch.sum((predicted_classes != target_classes) * target_classes == 1).float()
        tn += torch.sum((predicted_classes == target_classes) * (1 - target_classes) == 1).float()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1_score}')


def init_random_seed(value=const.SEED):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True
