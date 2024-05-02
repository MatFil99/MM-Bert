from collections import defaultdict
import copy 

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import get_scheduler, AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator
from tqdm.auto import tqdm
import evaluate
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedModel, BertConfig, BertPreTrainedModel

import cmbert
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds
# import multimodal_tokenizer

# def print_batch(batch):
#     print(f"input_ids shape: {batch['input_ids'].shape}")
#     print(f"audio_data shape: {batch['audio_data'].shape}")
#     print(f"visual_data shape: {batch['visual_data'].shape}")
#     print(f"labels shape: {batch['labels'].shape}")
#     print(f"last batch: {batch.keys()}")
#     print(f"labels cuda: {batch['labels'].is_cuda}")

class TrainingArgs():

    def __init__(self,
                 batch_size = 8,
                 num_epochs = 3,
                 criterion = 'crossentropy',
                 optimizer = 'adamw',
                 lr = 2e-5,
                 scheduler_type = 'linear',
                 num_warmup_steps = 0,
                 best_model_metric = 'accuracy',
                 best_model_path = None,
                 save_best_model = False
                 ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.best_model_metric = best_model_metric
        self.save_best_model = save_best_model
        self.best_model_path = best_model_path


class MultiModalDataCollator():

    def __init__(self, checkpoint, device=torch.device("cpu")) -> None:
        self.tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)
        self.device = device

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        rows = args[0]
        batch = {key: [row[key] for row in rows] for key in rows[0]}
        if 'audio_feat' not in batch:
            batch['audio_feat'] = None
        if 'visual_feat' not in batch:
            batch['visual_feat'] = None

        tokenized = self.tokenizer(text=batch['text_feat'], audio=batch['audio_feat'], visual=batch['visual_feat'], padding=True, truncation=True)
        tokenized['labels'] = torch.tensor(batch['labels'], dtype=torch.long)
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
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = cmbert.CmuDataset.from_dataset(ds, fold='train')
    test_ds = cmbert.CmuDataset.from_dataset(ds, fold='test')
    valid_ds = cmbert.CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds

def evaluation(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.eval()

    loss = 0
    metrics_names = ['accuracy', 'f1']
    metrics = {metric: evaluate.load(metric) for metric in metrics_names}

    with torch.no_grad():
        for batch in dataloader:
            # del vbatch['visual_data']
            # del vbatch['audio_data'] # COMMENT

            logits = model(**batch)['logits']
            loss += criterion(logits, batch['labels'])

            predictions = torch.argmax(logits, dim=-1)

            for _, metric in metrics.items():
                metric.add_batch(predictions=predictions, references=batch['labels'])

        calculated_metrics = {}
        for _, metric in metrics.items():
            calculated_metrics.update(metric.compute())
        calculated_metrics['loss'] = loss

    return calculated_metrics

def train(model, train_dataloader, valid_dataloader, num_epochs, optimizer, lr_scheduler, criterion):
    # num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []

    best_model = copy.deepcopy(model)
    best_accuracy = 0.0

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
        
        valid_evaluation = evaluation(model, valid_dataloader, criterion=criterion)
        print('==================================')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Training loss: {cum_loss}')
        print('Validation: ', valid_evaluation)
        if valid_evaluation['accuracy'] > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = valid_evaluation['accuracy']
        


    return best_model

def get_criterion(criterion_name):
    if criterion_name == 'crossentropyloss':
        return nn.CrossEntropyLoss()
    # elif criterion_name == '': # ...

def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name == 'adamw':
        return AdamW(**kwargs)

def main(dataset_config, model_config, training_arguments):
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # dataset
    ds = cmbert.CmuDataset(dataset_config)
    
    # model
    model = cmbert.CMBertForSequenceClassification(config=model_config)
    tokenizer = cmbert.MultimodalTokenizer(checkpoint=model_config.encoder_checkpoint)

    # split data into train, valid, test datasets
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds, num_labels=model_config.num_labels)

    batch_size = training_arguments.batch_size
    criterion = get_criterion(training_arguments.criterion)
    optimizer = get_optimizer(
        optimizer_name=training_arguments.optimizer, 
        params=model.parameters(), 
        lr=training_arguments.lr
        )
    
    model.to(device)
    criterion.to(device)

    data_collator = MultiModalDataCollator(checkpoint=model_config.encoder_checkpoint, device=device)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)

    num_epochs = training_arguments.num_epochs
    num_training_steps = num_epochs*len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=training_arguments.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_arguments.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_model = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
    )

    calculated_metrics = evaluation(best_model, test_dataloader)

    print(calculated_metrics)





if __name__ == '__main__':
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("train.py")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    load_preprocessed = True

    # dataset configuration
    if load_preprocessed:
        print('load_preprocessed')
        dataset_config = cmbert.CmuDatasetConfig(
            load_preprocessed=True,
            preprocess=False,    
            )
    else:
        dataset_config = cmbert.CmuDatasetConfig()


    # model configuration
    checkpoint = 'distilbert/distilbert-base-uncased'
    # checkpoint = 'google-bert/bert-base-uncased'
    model_config = cmbert.CMBertConfig(
        encoder_checkpoint=checkpoint,
        hidden_dropout_prob=0.1,
        hidden_size=768
    )
    num_labels = 2


    # create dataset, model, tokenizer
    
    if load_preprocessed:
        ds = cmbert.CmuDataset(dataset_config)
    else:
        ds = cmbert.CmuDataset(dataset_config)

    
    # bconfig = BertConfig.from_pretrained(checkpoint)
    model = cmbert.CMBertForSequenceClassification(config=model_config) #, num_labels=2)
    # model = cmbert.CMBertForSequenceClassification.from_pretrained(checkpoint, config=model_config, num_labels=2)
    tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)


    # save data
    # ds.deploy()
    
    # if load_preprocessed:
    #     ds.append_labels_to_dataset()
    
    

    # prepare data
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds)
    

    # # # 
    batch_size = 8
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=2e-5) # 1e-4 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    criterion.to(device)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_collator = MultiModalDataCollator(checkpoint=checkpoint, device=device)

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)

    num_epochs = 4
    num_training_steps = num_epochs*len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    best_model = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
    )

    calculated_metrics = evaluation(
        best_model, 
        test_dataloader, 
        criterion=criterion
        )

    print(calculated_metrics)
