import copy 
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate

from mmanalysis.datacollators import MMSeqClassDataCollator

from mmanalysis.datasets.cmu import (
    CmuDataset,
    standard_folds
)

# from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds
from mmanalysis.models.cmbert import (
    CMBertForSequenceClassification,
    CMBertTokenizer,
)

from mmanalysis.models.mmbert import (
    MMBertForSequenceClassification,
    MMBertTokenizer,
)

from mmanalysis.utils import (
    set_optimizer_custom_parameters,
    get_criterion,
    get_optimizer,
    save_result,
)


# def prepare_data_splits(ds, num_labels):
def prepare_data_splits(ds, num_labels):
    ds.labels_2_class(num_classes=num_labels)

    folds = {
        'train': standard_folds[ds.dsname].standard_train_fold,
        'valid': standard_folds[ds.dsname].standard_valid_fold,
        'test': standard_folds[ds.dsname].standard_test_fold,
        }
   
    ds.train_test_valid_split(folds=folds)
    ds.replace_inf_and_nan_values(value=0)
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = CmuDataset.from_dataset(ds, fold='train')
    test_ds = CmuDataset.from_dataset(ds, fold='test')
    valid_ds = CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds


def evaluation(model, dataloader, criterion, metrics_names, task):
    model.eval()

    loss = 0.0
    metrics = {metric: evaluate.load(metric) for metric in metrics_names}

    with torch.no_grad():
        for batch in dataloader:

            logits = model(**batch)['logits']
            loss += criterion(logits, batch['labels'])

            logitsnan = any(torch.isnan(logits))
            logitsinf = any(torch.isinf(logits))
            labelsnan = any(torch.isnan(batch["labels"]))
            labelsinf = any(torch.isinf(batch["labels"]))

            if logitsnan == True or logitsinf == True or labelsnan == True or labelsinf == True:
                print(f'logits nan: {any(torch.isnan(logits))} logits inf: {any(torch.isinf(logits))}')
                print(f'labels nan: {any(torch.isnan(batch["labels"]))} labels inf: {any(torch.isinf(batch["labels"]))}')
                print(f'batch {batch}')
                print(f'logits {logits}')

            for name, metric in metrics.items():
                metric.add_batch(predictions=logits, references=batch['labels'])

        calculated_metrics = {}
        for name, metric in metrics.items():
            metric_evaluation = metric.compute()
            calculated_metrics.update(metric_evaluation)

        calculated_metrics['loss'] = loss.item()

    return calculated_metrics

def train(model, train_dataloader, valid_dataloader, num_epochs, patience, optimizer, lr_scheduler, criterion, metrics, task, best_model_metric):
    # num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    valid_eval = []

    best_model = copy.deepcopy(model)
    best_eval = np.finfo(np.float32).max # 
    worse_count = 0

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
        if best_model_metric in ['mse', 'mae']: # adapt decision about best model
            best_coeff = -1
        else:
            best_coeff = 1

        if valid_evaluation[best_model_metric] * best_coeff > best_eval * best_coeff:
            best_model = copy.deepcopy(model)
            best_eval = valid_evaluation[best_model_metric]
            worse_count = 0
        else:
            worse_count += 1

        train_loss.append(cum_loss)
        valid_eval.append(valid_evaluation)

        if worse_count > patience:
            break

    return best_model, train_loss, valid_eval


def main(model_name, train_ds, valid_ds, test_ds, dataset_config, model_config, training_arguments, results_path='experiments/results.jsonl', pretrained_checkpoint=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
    
    # model
    if model_name == 'cmbert':
        ModelClass = CMBertForSequenceClassification
    elif model_name == 'mmbert':
        ModelClass = MMBertForSequenceClassification

    if pretrained_checkpoint is not None:
        model = ModelClass.from_pretrained(pretrained_checkpoint, 
                                           num_classes=model_config.num_classes)
        if model_config.freeze_params:
            if model_name == 'cmbert': # incompatibility, but pretrained models requires previous method names
                model.freeze_parameters()
            else:
                model.freeze_params()
        model.config = model_config
    else:
        model = ModelClass(config=model_config) 

    # task used for metric definition
    task = 'r'

    batch_size = training_arguments.batch_size
    criterion = get_criterion(training_arguments.criterion)

    # Prepare optimizer parameters
    if training_arguments.layer_specific_optimization:
        parameters = set_optimizer_custom_parameters(
            model=model,
        )
    else:
        parameters = model.parameters()


    optimizer = get_optimizer(
        optimizer_name=training_arguments.optimizer, 
        params=parameters,
        lr=training_arguments.lr
        )
    
    model.to(device)
    criterion.to(device)

    
    # tokenizer
    if model_name == 'cmbert':
        TokenizerClass = CMBertTokenizer
    elif model_name == 'mmbert':
        TokenizerClass = MMBertTokenizer

    tokenizer = TokenizerClass(checkpoint=model_config.encoder_checkpoint)

    data_collator = MMSeqClassDataCollator(tokenizer=tokenizer, num_labels=model_config.num_classes, device=device)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_ds, batch_size=1, collate_fn=data_collator) # batch_size for testing = 1
    
    # regression
    metrics = ['mse', 'mae', 'pearsonr']

    num_epochs = training_arguments.num_epochs
    num_training_steps = num_epochs*len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=training_arguments.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_arguments.warmup_steps_ratio * num_training_steps,
        num_training_steps=num_training_steps,
    )

    best_model, train_loss, valid_eval = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        patience=training_arguments.patience,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        metrics=metrics,
        task=task,
        best_model_metric=model_config.best_model_metric,
    )

    datetime_run = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    model_checkpoint_name = model_config.encoder_checkpoint.split('/')[-1] + '_' + datetime_run
    full_path = training_arguments.save_model_dest + '/' + model_name + '_' + model_checkpoint_name
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
        'datetime_run': datetime_run,
        'dataset_config': dataset_config.__dict__,
        'training_arguments': training_arguments.__dict__,
        'model_name': model_name,
        'model_config': model_config.__dict__,
        'train_loss': train_loss,
        'valid_eval': valid_eval,
        'test_eval': calculated_metrics,
    }

    save_result(result, results_path)
