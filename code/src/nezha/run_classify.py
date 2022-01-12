# coding:utf-8

import gc
import csv
import sys
import warnings
import pandas as pd
from torch import multiprocessing
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold

from src.nezha.util.classifier_utils import *
from src.nezha.modeling.modeling import NeZhaModel, NeZhaPreTrainedModel

sys.path.append('../../src')
multiprocessing.set_sharing_strategy('file_system')


class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 25
        self.bert = NeZhaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        attention_mask = torch.ne(input_ids, 0)

        encoder_out, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.classifier(pooled_out)
        outputs = (logits,) + (pooled_out,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def read_data(args, tokenizer):
    train_inputs = defaultdict(list)
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            id, name, content, label = line.strip().split(',')
            if str(name) == 'nan':
                name = '无'
            if str(content) == 'nan':
                content = '无'
            label = int(label)
            build_bert_inputs(train_inputs, label, name, tokenizer, content)

    test_inputs = defaultdict(list)
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            id, name, content = line.strip().split(',')
            if str(name) == 'nan':
                name = '无'
            if str(content) == 'nan':
                content = '无'
            build_bert_inputs_test(test_inputs, name, tokenizer, content)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    test_cache_pkl_path = os.path.join(args.data_cache_path, 'test.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(test_inputs, test_cache_pkl_path)


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = NeZhaForSequenceClassification.from_pretrained(args.model_path)
    model.to(args.device)

    return tokenizer, model


def train(args):
    tokenizer, model = build_model_and_tokenizer(args)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer)

    train_dataloader, test_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Training epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

    save_model(model, tokenizer, args.output_path)

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def train_cv(args):
    skf = StratifiedKFold(shuffle=True, n_splits=args.num_folds)

    train = pd.read_csv(args.train_path, sep=',')
    y = train.iloc[:, 3]

    for fold, (train_index, dev_index) in enumerate(skf.split(X=range(train.shape[0]), y=y)):

        print(f'>>> this is fold {fold} .')

        tokenizer, model = build_model_and_tokenizer(args)

        if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
            read_data(args, tokenizer)

        train_dataloader, val_dataloader = load_cv_data(args, train_index, dev_index, tokenizer)

        total_steps = args.num_epochs * len(train_dataloader)

        optimizer, scheduler = build_optimizer(args, model, total_steps)

        total_loss, cur_avg_loss, global_steps = 0., 0., 0
        best_acc_score = 0.

        for epoch in range(1, args.num_epochs + 1):

            train_iterator = tqdm(train_dataloader, desc=f'Training epoch : {epoch}', total=len(train_dataloader))

            model.train()

            for batch in train_iterator:
                batch_cuda = batch2cuda(args, batch)
                loss, logits = model(**batch_cuda)[:2]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                total_loss += loss.item()
                cur_avg_loss += loss.item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

                if (global_steps + 1) % args.logging_step == 0:

                    epoch_avg_loss = cur_avg_loss / args.logging_step
                    global_avg_loss = total_loss / (global_steps + 1)

                    print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                          f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                    metric = evaluation(args, model, val_dataloader)
                    acc, f1, avg_val_loss = metric['acc'], metric['f1'], metric['avg_val_loss']

                    if acc > best_acc_score:
                        best_acc_score = acc
                        model_save_path = os.path.join(args.output_path, f'cv/last-checkpoint-{fold}')
                        save_model(model, tokenizer, model_save_path)

                        print(f'\n>>>\n    best acc - {best_acc_score}, '
                              f'dev loss - {avg_val_loss} .')

                    model.train()

                    cur_avg_loss = 0.0
                global_steps += 1

        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()


def predict_full(args):

    print('\n>> loading test dataset ... ...')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_dataloader, test_dataloader = load_data(args, tokenizer)

    print('\n>> full data')

    print('\n>> start predicting ... ...')
    best_model = NeZhaForSequenceClassification.from_pretrained(args.output_path)
    best_model.to(args.device)
    best_model.eval()

    val_iterator = tqdm(test_dataloader, desc='Predict test data',
                        total=len(test_dataloader))
    p_logit = []
    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            logits = best_model(**batch_cuda)[0]
            p_logit.extend(torch.softmax(logits, -1).cpu().numpy())

    res = np.vstack(p_logit)

    final_res = res
    final_res.tolist()

    predictions = np.argmax(final_res, axis=-1)

    k = 0
    result = []
    for i in predictions:
        result.append((k, str(i)))
        k += 1

    with open(args.submit_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerow(['id', 'label'])
        tsv_w.writerows(result)

    print('\n>> predict completed .')


def predict_cv(args):

    print('\n>> loading test dataset ... ...')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_dataloader, test_dataloader = load_data(args, tokenizer)

    print('\n>> cv data')

    print('\n>> start predicting ... ...')

    final_res = None
    for fold in range(args.num_folds):
        best_model_path = os.path.join(args.output_path, f'cv/last-checkpoint-{fold}')
        best_model = NeZhaForSequenceClassification.from_pretrained(best_model_path)
        best_model.to(args.device)
        best_model.eval()

        test_iterator = tqdm(test_dataloader, desc='Predict test data', total=len(test_dataloader))

        p_logit = []
        with torch.no_grad():
            for batch in test_iterator:
                batch_cuda = batch2cuda(args, batch)
                logits = best_model(**batch_cuda)[0]
                p_logit.extend(torch.softmax(logits, -1).cpu().numpy())

        res = np.vstack(p_logit)

        if final_res is None:
            final_res = res
        else:
            final_res += res

    final_res /= args.num_folds

    final_res.tolist()

    print('\n>> combining ... ...')

    predictions = np.argmax(final_res, axis=-1)

    k = 0
    result = []
    for i in predictions:
        result.append((k, str(i)))
        k += 1

    with open(args.submit_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerow(['id', 'label'])
        tsv_w.writerows(result)

    print('\n>> predict completed .')


def main(task_type):
    parser = ArgumentParser()

    parser.add_argument('--output_path', type=str,
                        default=f'../../user_data/nezha/{task_type}/output_model/nezha')
    parser.add_argument('--submit_path', type=str,
                        default='../../submita.csv')
    parser.add_argument('--train_path', type=str,
                        default=f'../../data/{task_type}/train.csv')
    parser.add_argument('--test_path', type=str,
                        default=f'../../data/{task_type}/testa_nolabel.csv')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../../user_data/nezha/process_data/pkl/{task_type}/')

    parser.add_argument('--model_path', type=str,
                        default=f'../../user_data/self_pretrained_model/')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=350)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logging_step', type=int, default=300)

    parser.add_argument('--seed', type=int, default=9527)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--cv', type=bool, default=True)
    parser.add_argument('--num_folds', type=int, default=5)

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    make_dirs(path_list)

    seed_everything(args.seed)

    if args.cv:
        train_cv(args)
        predict_cv(args)
    else:
        train(args)
        predict_full(args)


if __name__ == '__main__':
    main('shandong')
