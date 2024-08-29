import numpy as np
import copy 
import math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate


from mmanalysis.models.cmbert import (
    # CMBertConfig,
    CMBertForMaskedLM,
    CMBertTokenizer,
)
from mmanalysis.models.mmbert import (
    MMBertConfig,
    MMBertForMaskedLM,
    MMBertTokenizer,
)

from mmanalysis.datacollators import MMMLMDataCollator

from mmanalysis.datasets.cmu import (
    CmuDataset,
    standard_folds,
    SDK_DS,
    CmuDatasetConfig
)

from mmanalysis.utils import (
    get_criterion,
    get_optimizer,
    group_texts,
    save_result,
    set_optimizer_custom_parameters
)

def prepare_data_splits(ds, *args):
    ds.features_as_dtype(np.float32)

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


def tokenize_function(batch, *args, **kwargs):
    if 'audio_feat' in batch:
        audio = [np.array(element) for element in batch['audio_feat']]
    else:
        audio = None
    if 'visual_feat' in batch:
        visual = [np.array(element) for element in batch['visual_feat']]
    else:
        visual = None
    
    tokenizer = kwargs['tokenizer']
    tokenized = tokenizer(text=batch['text_feat'], audio=audio, visual=visual, padding=False, return_tensors=None)
    if tokenizer.text_tokenizer.is_fast:
        tokenized["word_ids"] = [np.array(tokenized.word_ids(i)) for i in range(len(tokenized["input_ids"]))]

    return tokenized

def evaluation(model, dataloader, metrics_names=['accuracy', 'f1', 'precision', 'recall']):
    model.eval()
    losses = []

    # TEST adding extra evaluation metrics for mlm task
    # metrics = {metric: evaluate.load(metric) for metric in metrics_names}
    # TEST

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.unsqueeze(0))

    # TEST
    #     predictions = torch.argmax(outputs.logits, dim=-1)
    #     for name, metric in metrics.items():
    #         metric.add_batch(predictions=predictions, references=batch['labels'])
    # calculated_metrics = {}
    # for name, metric in metrics.items():
    #     # if neighter binary classification nor regression
    #     if name in ['f1', 'precision', 'recall']:
    #         metric_evaluation = metric.compute(average='weighted')
    #     else:
    #             metric_evaluation = metric.compute()

    #     calculated_metrics.update(metric_evaluation)
    # TEST

    losses = torch.tensor(losses[: len(dataloader)])
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    # TEST
    # calculated_metrics['perplexity'] = perplexity
    # calculated_metrics['loss'] = torch.sum(losses).item()
    # return calculated_metrics
    # TEST

    return {'perplexity': perplexity, 'loss': torch.sum(losses).item()}


# def train(num_epochs,):
def train(model, train_dataloader, valid_dataloader, num_epochs, patience, optimizer, lr_scheduler, criterion=None, metrics=None, task=None, best_model_metric='perplexity'):
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    valid_eval = []
    best_eval = np.finfo(np.float32).max # 
    best_model = model
    worse_count = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        cum_loss = 0.0
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            cum_loss += loss.item()

        # Evaluation
        valid_evaluation = evaluation(
            model=model,
            dataloader=valid_dataloader,
        )

        print(f"====================================")
        print(f"Epoch {epoch} / {num_epochs}")
        print(f"Training loss: {cum_loss}")
        print(f"Validation: {valid_evaluation}")
        if valid_evaluation[best_model_metric] < best_eval:
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

# def main(model_name, dataset_config, model_config, training_arguments, results_path='experiments/results.jsonl', dsdeploy=False):
def main(model_name, train_ds, valid_ds, test_ds, dataset_config, model_config, training_arguments, results_path='experiments/results.jsonl', pretrained_checkpoint=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)

    if model_name == 'cmbert':
        TokenizerClass = CMBertTokenizer
    elif model_name == 'mmbert':
        TokenizerClass = MMBertTokenizer

    tokenizer = TokenizerClass(checkpoint=model_config.encoder_checkpoint)

    dsdict = DatasetDict({
        'train': Dataset.from_dict(train_ds.dataset),
        'valid': Dataset.from_dict(valid_ds.dataset),
        'test': Dataset.from_dict(test_ds.dataset),
    })

    remove_columns = []
    if 'text_feat' in dataset_config.feature_names:
        remove_columns.append('text_feat')
    if 'audio_feat' in dataset_config.feature_names:
        remove_columns.append('audio_feat')
    if 'visual_feat' in dataset_config.feature_names:
        remove_columns.append('visual_feat')

    tokenized_datasets = dsdict.map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_columns,
        fn_kwargs={'tokenizer': tokenizer}
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        fn_kwargs={'chunk_size': training_arguments.chunk_size}
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_collator = MMMLMDataCollator(tokenizer=tokenizer, wwm_probability=training_arguments.wwm_probability, return_tensors='pt', device=device)
    
    batch_size = training_arguments.batch_size
  
    train_dataloader = DataLoader(lm_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(lm_datasets['valid'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(lm_datasets['test'], batch_size=batch_size, collate_fn=data_collator)

    # model
    if model_name == 'cmbert':
        ModelClass = CMBertForMaskedLM
    elif model_name == 'mmbert':
        ModelClass = MMBertForMaskedLM

    model = ModelClass(model_config)
    model.to(device)

    # Prepare optimizer
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

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = training_arguments.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_model, train_loss, valid_eval = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=training_arguments.num_epochs,
        patience=training_arguments.patience,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    datetime_run = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    model_checkpoint_name = model_config.encoder_checkpoint.split('/')[-1] + '_' + datetime_run
    full_path = training_arguments.save_model_dest + '/' + model_name + '_' + model_checkpoint_name
    if training_arguments.save_best_model:
        best_model.save_pretrained(
            save_directory=full_path,
            state_dict=best_model.state_dict(),
        )

    test_eval = evaluation(
        model=best_model,
        dataloader=test_dataloader,
    )

    print(test_eval)

    result = {
        'datetime_run': datetime_run,
        'dataset_config': dataset_config.__dict__,
        'training_arguments': training_arguments.__dict__,
        'model_config': model_config.__dict__,
        'model_name': model_name,
        'train_loss': train_loss,
        'valid_eval': valid_eval,
        'test_eval': test_eval,
    }

    save_result(result, results_path)






