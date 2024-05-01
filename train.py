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

def print_batch(batch):
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"audio_data shape: {batch['audio_data'].shape}")
    print(f"visual_data shape: {batch['visual_data'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"last batch: {batch.keys()}")
    print(f"labels cuda: {batch['labels'].is_cuda}")


class MultiModalDataCollator():

    def __init__(self, checkpoint, device=torch.device("cpu")) -> None:
        self.tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)
        self.device = device

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        rows = args[0]
        batch = {key: [row[key] for row in rows] for key in rows[0]}
        
        tokenized = self.tokenizer(text=batch['text_feat'], audio=batch['audio_feat'], visual=batch['visual_feat'], padding=True, truncation=True)
        tokenized['labels'] = torch.tensor(batch['labels'], dtype=torch.long)
        tokenized.to(self.device)
        return  tokenized

def prepare_data_splits(ds):
    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold,
        # 'train': ['03bSnISJMiM', '0h-zjBukYpk', '1DmNV9C1hbY', '1iG0909rllw', '2WGyTLYerpo'], # standard_folds.standard_train_fold,
        # 'valid': ['2iD-tVS8NPw', '5W7Z1C_fDaE', '6Egk_28TtTM'], # standard_folds.standard_valid_fold,
        # 'test': ['6_0THN4chvY', '73jzhE8R1TQ']# standard_folds.standard_test_fold
        }
   
    ds.train_test_valid_split(folds=folds)
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = cmbert.CmuDataset.from_dataset(ds, fold='train')
    test_ds = cmbert.CmuDataset.from_dataset(ds, fold='test')
    valid_ds = cmbert.CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds

def evaluation(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    vloss = 0

#
    metrics_names = ['accuracy', 'f1']
    metrics = {metric: evaluate.load(metric) for metric in metrics_names}
#

    with torch.no_grad():
        for vbatch in dataloader:
            # del vbatch['visual_data']
            # del vbatch['audio_data'] # COMMENT

            logits = model(**vbatch)['logits']
            vloss += criterion(logits, vbatch['labels'])
            total_acc += (logits.argmax(1) == vbatch['labels']).sum().item()
            total_count += len(vbatch['labels'])

            predictions = torch.argmax(logits, dim=-1)

#
            for name, metric in metrics.items():
                metric.add_batch(predictions=predictions, references=vbatch['labels'])
#    

        calculated_metrics = {}
        for _, metric in metrics.items():
            calculated_metrics.update(metric.compute())
        calculated_metrics['loss'] = vloss
        calculated_metrics['accuracy_0'] = total_acc / total_count

    return calculated_metrics
    # return {'accuracy': total_acc / total_count}


    # metrics_names = ['accuracy', 'f1']
    # metrics = {metric: evaluate.load(metric) for metric in metrics_names}

    # model.eval()
    # vloss = 0.0
    # with torch.no_grad():
    #     for vbatch in dataloader:
    #         del vbatch['visual_data']
    #         del vbatch['audio_data'] # COMMENT

    #         # logits, text_att, fusion_att = model(**batch) # UNCOMMENT
    #         logits = model(**vbatch)['logits']

    #         vloss += criterion(logits, vbatch['labels']).item()

    #         predictions = torch.argmax(logits, dim=-1)

    #         for name, metric in metrics.items():
    #             metric.add_batch(predictions=predictions, references=vbatch['labels'])
        
    #         optimizer.zero_grad()

    # calculated_metrics = {}
    # for _, metric in metrics.items():
    #     calculated_metrics.update(metric.compute())
    # calculated_metrics['loss'] = vloss

    # return calculated_metrics


def train(model, train_dataloader, valid_dataloader, num_epochs, optimizer, lr_scheduler, criterion):
    # num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []

    best_model = copy.deepcopy(model)
    best_accuracy = 0.0
    
    params_prev = None
    params_curr = None

    for epoch in range(num_epochs):
        
        model.train()
        # prev_model = copy.deepcopy(model)
        # params_prev = prev_model.parameters()

        with open('parameters' + str(epoch) + '.txt', 'w+') as fd:
            for params in model.parameters():
                fd.write(str(params))
                # print(param)
        cum_loss = 0.0
        for batch in train_dataloader:
            # del batch['visual_data']
            # del batch['audio_data']

            # logits, _, _ = model(**batch)
            output = model(**batch)
            logits = output['logits']

            loss = criterion(logits, batch['labels'])
            loss.backward()
            cum_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        valid_evaluation = evaluation(model, valid_dataloader)
        print('==================================')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Training loss: {cum_loss}')
        print('Validation: ', valid_evaluation)
        if valid_evaluation['accuracy'] > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = valid_evaluation['accuracy']
        
        # params_curr = model.parameters()
        # with open('cm_deltaparameters' + str(epoch) + '.txt', 'w+') as fd:
        #     if params_prev:
        #         for param_prev, param_curr in zip(params_prev, params_curr):
        #             fd.write(str(param_curr - param_prev))
                    # print(param)
        

    return best_model

if __name__ == '__main__':
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("train.py")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    load_preprocessed = True

    # dataset configuration
    if load_preprocessed:
        dataset_config = cmbert.DataConfig(
            ds_path=r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK\cmumosi\aligned',        
            load_preprocessed=True        
            )
    else:
        dataset_config = cmbert.DataConfig()

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
        ds = cmbert.CmuDataset(dataset_config, preprocess=False)
    else:
        ds = cmbert.CmuDataset(dataset_config, preprocess=True)

    # bconfig = BertConfig.from_pretrained(checkpoint)
    model = cmbert.CMBertForSequenceClassification(config=model_config) #, num_labels=2)
    # model = cmbert.CMBertForSequenceClassification.from_pretrained(checkpoint, config=model_config, num_labels=2)
    tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)


    # save data
    # ds.deploy()
    
    if load_preprocessed:
        ds.append_labels_to_dataset()
    
    ds.labels_2_class(num_classes=num_labels)

    # prepare data
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds)
    

    # from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
    
    # model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)


    # # # 
    batch_size = 8
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5) # 1e-4 
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

    # print(len(train_dataloader))

    # i = 0
    # for batch in train_dataloader:
        # del batch['visual_data']
        # logits, text_att, fusion_att = model(**batch)
        # print(f'logits: {logits}')
        # print(f'text_att: {text_att.shape}')
        # print(f'fusion_att: {fusion_att.shape}')

    
        
    #     # break
    # # print(i)
    # # print(sentences)




    
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
