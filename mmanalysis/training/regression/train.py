import copy 

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

            for name, metric in metrics.items():
                metric.add_batch(predictions=logits, references=batch['labels'])

        calculated_metrics = {}
        for name, metric in metrics.items():
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


def main(dataset_config, model_config, training_arguments, results_path='experiments/results.jsonl', dsdeploy=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
    

    # dataset
    ds = CmuDataset(dataset_config)
    if dsdeploy:
        ds.deploy()
    
    # model
    model = CMBertForSequenceClassification(config=model_config)
    # model = MMBertForSequenceClassification(config=model_config)
    
    # task used for metric definition
    task = 'r'

    # split data into train, valid, test datasets
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds, num_labels=model_config.num_classes)

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

    # tokenizer = MMBertTokenizer(checkpoint=model_config.encoder_checkpoint)
    tokenizer = CMBertTokenizer(checkpoint=model_config.encoder_checkpoint)

    data_collator = MMSeqClassDataCollator(tokenizer=tokenizer, num_labels=model_config.num_classes, device=device)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
    
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
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        metrics=metrics,
        task=task,
        best_model_metric=model_config.best_model_metric,
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
