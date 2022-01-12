# coding:utf-8

import re
import os
import torch
import pickle
import random
import shutil
import scipy.stats
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(42)


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def save_model(model, tokenizer, saving_path):
    os.makedirs(saving_path, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(saving_path)
    tokenizer.save_vocabulary(saving_path)


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def build_bert_inputs(inputs, label, sentence, tokenizer, sentence_b=None):
    if sentence_b is not None:
        inputs_dict = tokenizer.encode_plus(sentence, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
    else:
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    inputs['input_ids'].append(input_ids)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['labels'].append(label)


def build_bert_inputs_test(inputs, sentence, tokenizer, sentence_b=None):
    if sentence_b is not None:
        inputs_dict = tokenizer.encode_plus(sentence, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
    else:
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    inputs['input_ids'].append(input_ids)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)


class KGDataset(Dataset):
    def __init__(self, data_dict: dict, tokenizer: BertTokenizer):
        super(KGDataset, self).__init__()
        self.data_dict = data_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class KGDataset_test(Dataset):
    def __init__(self, data_dict: dict, tokenizer: BertTokenizer):
        super(KGDataset_test, self).__init__()
        self.data_dict = data_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, labels_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])

            # pad
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

            # cut
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        labels = torch.tensor(labels_list, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list, labels_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


class Collator_test:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])

            # pad
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

            # cut
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }

        return data_dict


def load_data(args, tokenizer):
    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    test_cache_pkl_path = os.path.join(args.data_cache_path, 'test.pkl')

    with open(train_cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_cache_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)
    collate_fn_test = Collator_test(args.max_seq_len, tokenizer)

    train_dataset = KGDataset(train_data, tokenizer)
    test_dataset = KGDataset_test(test_data, tokenizer)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size * 8, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn_test)

    return train_dataloader, test_dataloader


def load_cv_data(args, train_index, dev_index, tokenizer):
    train_data = defaultdict(list)
    dev_data = defaultdict(list)

    cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')

    with open(cache_pkl_path, 'rb') as f:
        all_data = pickle.load(f)

    for key, values in all_data.items():
        train_data[key] = [values[idx] for idx in train_index]
        dev_data[key] = [values[idx] for idx in dev_index]

    collate_fn = Collator(args.max_seq_len, tokenizer)

    train_dataset = KGDataset(train_data, tokenizer)
    dev_dataset = KGDataset(dev_data, tokenizer)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_fn)

    return train_dataloader, dev_dataloader


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=args.learning_rate, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def evaluation(args, model, val_dataloader):
    model.eval()

    metric = {}
    preds, labels = [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            val_loss += loss.item()

            preds.extend([i for i in torch.argmax(torch.softmax(logits, -1), 1).cpu().numpy().tolist()])
            labels.extend([i for i in batch_cuda['labels'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    acc, f1 = accuracy_score(y_true=labels, y_pred=preds), f1_score(y_true=labels, y_pred=preds, average='macro')
    avg_val_loss, acc, f1 = round(avg_val_loss, 4), round(acc, 4), round(f1, 4)

    metric['acc'], metric['f1'], metric['avg_val_loss'] = acc, f1, avg_val_loss

    return metric


def make_dirs(path_list):
    for i in path_list:
        os.makedirs(os.path.dirname(i), exist_ok=True)
