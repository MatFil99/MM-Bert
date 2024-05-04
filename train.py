from collections import defaultdict
import copy 
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam, RMSprop, Adadelta
import transformers
from transformers import get_scheduler, AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator
from tqdm.auto import tqdm
import evaluate
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedModel, BertConfig, BertPreTrainedModel

import cmbert
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds

class TrainingArgs():

    def __init__(self,
                 batch_size = 8,
                 num_epochs = 3,
                 criterion = 'crossentropy',
                 optimizer = 'adamw',
                 lr = 2e-5,
                 scheduler_type = 'linear',
                 warmup_steps_ratio = 0.0,
                 best_model_metric = 'accuracy',
                 best_model_path = None,
                 save_best_model = False,
                 save_model_dest = None,
                 ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps_ratio = warmup_steps_ratio
        self.best_model_metric = best_model_metric
        self.save_best_model = save_best_model
        self.best_model_path = best_model_path
        self.save_model_dest = save_model_dest

    def __str__(self) -> str:
        return str(self.__dict__.items())

class MultiModalDataCollator():

    def __init__(self, checkpoint, num_labels=2 ,device=torch.device("cpu")) -> None:
        self.tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)
        self.device = device
        if num_labels == 1:
            self.labels_dtype = torch.float32
        else:
            self.labels_dtype = torch.long

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        rows = args[0]
        batch = {key: [row[key] for row in rows] for key in rows[0]}
        if 'audio_feat' not in batch:
            batch['audio_feat'] = None
        if 'visual_feat' not in batch:
            batch['visual_feat'] = None

        tokenized = self.tokenizer(text=batch['text_feat'], audio=batch['audio_feat'], visual=batch['visual_feat'], padding=True, truncation=True)
        tokenized['labels'] = torch.tensor(batch['labels'], dtype=self.labels_dtype)
        tokenized.to(self.device)
        return  tokenized

def prepare_data_splits(ds, num_labels):
    ds.labels_2_class(num_classes=num_labels)

    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold,
        }
   
    ds.train_test_valid_split(folds=folds)
    ds.replace_inf_and_nan_values(value=0)
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = cmbert.CmuDataset.from_dataset(ds, fold='train')
    test_ds = cmbert.CmuDataset.from_dataset(ds, fold='test')
    valid_ds = cmbert.CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds


from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def evaluation(model, dataloader, criterion, metrics_names, task):
    model.eval()

    loss = 0
    metrics = {metric: evaluate.load(metric) for metric in metrics_names}

    with torch.no_grad():
        for batch in dataloader:

            logits = model(**batch)['logits']
            loss += criterion(logits, batch['labels'])

            if 'accuracy' in metrics_names:
                predictions = torch.argmax(logits, dim=-1)
            else:
                predictions = logits


            for name, metric in metrics.items():
                metric.add_batch(predictions=predictions, references=batch['labels'])

        calculated_metrics = {}
        for name, metric in metrics.items():
            # if neighter binary classification nor regression
            if task not in ['c2', 'r'] and name in ['f1']:
                metric_evaluation = metric.compute(average='weighted')
            else:
                metric_evaluation = metric.compute()
            calculated_metrics.update(metric_evaluation)

        calculated_metrics['loss'] = loss.item()

    return calculated_metrics

def train(model, train_dataloader, valid_dataloader, num_epochs, optimizer, lr_scheduler, criterion, metrics, task, best_model_metric):
    # num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    valid_eval = []

    best_model = copy.deepcopy(model)
    best_eval = 0.0

    for epoch in range(num_epochs):
        
        model.train()
        cum_loss = 0.0
        for batch in train_dataloader:
            
            output = model(**batch)
            logits = output['logits']

            loss = criterion(logits, batch['labels'])
            loss.backward()
            cum_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        valid_evaluation = evaluation(
            model, 
            valid_dataloader, 
            criterion=criterion, 
            metrics_names=metrics,
            task=task,
            )
        print('==================================')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Training loss: {cum_loss}')
        print('Validation: ', valid_evaluation)
        if valid_evaluation[best_model_metric] > best_eval:
            best_model = copy.deepcopy(model)
            best_eval = valid_evaluation[best_model_metric]

        train_loss.append(cum_loss)
        valid_eval.append(valid_evaluation)

    return best_model, train_loss, valid_eval

def get_criterion(criterion_name):
    if criterion_name == 'crossentropyloss':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'mseloss': # ...
        return nn.MSELoss()
    elif criterion_name == 'maeloss':
        return nn.L1Loss()
    

def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name == 'adamw':
        return AdamW(**kwargs)
    elif optimizer_name == 'adam':
        return Adam(**kwargs)
    elif optimizer_name == 'rmsprop':
        return RMSprop(**kwargs)
    elif optimizer_name == 'adadelta':
        return Adadelta(**kwargs)

def save_result(result, path):
    with open(path, 'a+') as fd:
        json.dump(result, fd)
        fd.write('\n')
        

def main(dataset_config, model_config, training_arguments, results_path='experiments/results.jsonl', dsdeploy=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
    

    # dataset
    ds = cmbert.CmuDataset(dataset_config)
    if dsdeploy:
        ds.deploy()
    
    # model
    model = cmbert.CMBertForSequenceClassification(config=model_config)
    if model_config.num_classes == 1:
        task = 'r'
    elif model_config.num_classes == 2:
        task = 'c2'
    elif model_config.num_classes == 7:
        task = 'c7'

    # split data into train, valid, test datasets
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds, num_labels=model_config.num_classes)

    batch_size = training_arguments.batch_size
    criterion = get_criterion(training_arguments.criterion)

    # Prepare optimizer
    if ('audio_feat' in dataset_config.feature_names 
        or 'visual_feat' in dataset_config.feature_names):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight']
        new_decay = ['modality_fusion']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(np in n for np in new_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay )and any(np in n for np in new_decay)],'lr':0.01}
        ]

        parameters = optimizer_grouped_parameters
    else:
        parameters = model.parameters()


    optimizer = get_optimizer(
        optimizer_name=training_arguments.optimizer, 
        params=parameters,
        lr=training_arguments.lr
        )
    
    model.to(device)
    criterion.to(device)

    data_collator = MultiModalDataCollator(checkpoint=model_config.encoder_checkpoint, num_labels=model_config.num_classes, device=device)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
    

    if model_config.num_classes == 1: # regression
        metrics = ['mse', 'mae', 'pearsonr']
    else: # classification
        metrics = ['accuracy', 'f1']


    num_epochs = training_arguments.num_epochs
    num_training_steps = num_epochs*len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=training_arguments.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_arguments.warmup_steps_ratio * num_training_steps,
        num_training_steps=num_training_steps,
    )

    # print(model_config)

    best_model, train_loss, valid_eval = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        metrics=metrics,
        task=task,
        best_model_metric=model_config.best_model_metric,
        # save_best_model=training_arguments.save_best_model,
    )

    best_model = model

    full_path = training_arguments.save_model_dest + '/' + model_config.encoder_checkpoint.split('/')[-1]
    
    if training_arguments.save_best_model:
        best_model.save_pretrained(
            save_directory=full_path,
            state_dict=best_model.state_dict(),
        )

    calculated_metrics = evaluation(
        model=best_model, 
        dataloader=test_dataloader,
        criterion=criterion,
        metrics_names=metrics,
        task=task,
        )

    print(calculated_metrics)

    result = {
        'dataset_config': dataset_config.__dict__,
        'training_arguments': training_arguments.__dict__,
        'model_config': model_config.__dict__,
        'train_loss': train_loss,
        'valid_eval': valid_eval,
        'test_eval': calculated_metrics,
    }

    save_result(result, results_path)

    # print(f'text_weight_1: {best_model.modality_fusion.text_weight_1}')
    # print(f'audio_weight_1: {best_model.modality_fusion.audio_weight_1}')
    # print(f'visual_weight_1: {best_model.modality_fusion.visual_weight_1}')