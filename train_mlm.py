from collections import defaultdict
import copy 
import json
import math
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam, RMSprop, Adadelta
import transformers
from transformers import get_scheduler, AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator
from tqdm.auto import tqdm
import evaluate
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedModel, BertConfig, BertPreTrainedModel, AutoModelForMaskedLM

import model_config
import cmbert.masked_language_modeling as mlm

import cmbert
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds
from cmbert import SDK_DS


import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2

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


class MLMDataCollator():

    def __init__(self, checkpoint, return_tensors='list', device=torch.device('cpu')) -> None:
        self.tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)
        self.device = device
        self.return_tensors = return_tensors

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        rows = args[0]
        batch = {key: [row[key] for row in rows] for key in rows[0]}


        features = []
        for i in range(len(batch['input_ids'])):
            features.append({})
            for k, v in batch.items():
                features[i][k] = v[i]     

        batch = self.whole_word_masking_data_collator(batch)
    
        return batch


    def whole_word_masking_data_collator(self, batch):
        tokenizer = self.tokenizer

        word_ids_list = batch.pop('word_ids')
        labels_list = batch['labels']

        for i in range(len(batch['input_ids'])):
            word_ids = word_ids_list[i]
            labels = labels_list[i]

            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            np.random.seed(7)
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    batch['input_ids'][i][idx] = tokenizer.mask_token_id
            batch["labels"][i] = new_labels

        for key, value in batch.items():
            if self.return_tensors == 'pt':
                if key == 'labels':
                    batch[key] = torch.tensor(value, dtype=torch.long).to(self.device)
                else:
                    batch[key] = torch.tensor(value).to(self.device)
            elif self.return_tensors == 'np':
                batch[key] = np.array(value)

        return batch
     

    def whole_word_masking_data_collator_bck(self, features):
        tokenizer = self.tokenizer
        for feature in features:
            for key in feature:
                feature[key] = np.array(feature[key])

            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            np.random.seed(7)
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels

        return features

def tokenize_function(batch, *args, **kwargs):
    audio = [np.array(element) for element in batch['audio_feat']]
    visual = [np.array(element) for element in batch['visual_feat']]
    
    tokenizer = kwargs['tokenizer']
    tokenized = tokenizer(text=batch['text_feat'], audio=audio, visual=visual, padding=False, return_tensors=None)
    if tokenizer.text_tokenizer.is_fast:
        tokenized["word_ids"] = [np.array(tokenized.word_ids(i)) for i in range(len(tokenized["input_ids"]))]

    return tokenized

def group_texts(examples):
    chunk_size = 128

    # # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples['input_ids'])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()

    return result

def save_result(result, path):
    with open(path, 'a+') as fd:
        json.dump(result, fd)
        fd.write('\n')

def evaluation(model, dataloader, metrics_names=None):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.unsqueeze(0))

    # losses = torch.cat(losses)
    losses = torch.tensor(losses[: len(dataloader)])
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    return {'perplexity': perplexity, 'loss': torch.sum(losses).item()}


# def train(num_epochs,):
def train(model, train_dataloader, valid_dataloader, num_epochs, optimizer, lr_scheduler, criterion=None, metrics=None, task=None, best_model_metric='perplexity'):
    pass
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    valid_eval = []
    best_eval = np.finfo(np.float32).max # 
    best_model = model

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

        train_loss.append(cum_loss)
        valid_eval.append(valid_evaluation)
    
    return best_model, train_loss, valid_eval

def main(dataset_config, _model_config, training_arguments, results_path='experiments/results.jsonl', dsdeploy=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)

    tokenizer = cmbert.MultimodalTokenizer(checkpoint=model_config.encoder_checkpoint)

    # dataset
    ds = cmbert.CmuDataset(dataset_config)
    if dsdeploy:
        ds.deploy()

    # model

    # split data into train, valid, test datasets
    train_ds, valid_ds, test_ds = mlm.prepare_data_splits(ds=ds, num_labels=_model_config.num_classes)

    dsdict = mlm.cmudataset_to_datasetdict(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        )

    tokenized_datasets = dsdict.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text_feat', 'audio_feat', 'visual_feat'],
        fn_kwargs={'tokenizer': tokenizer}
    )

    lm_datasets = tokenized_datasets.map(
        group_texts, 
        batched=True
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_collator = MLMDataCollator(_model_config.encoder_checkpoint, return_tensors='pt', device=device)
    # data_collator = MLMDataCollator(_model_config.encoder_checkpoint, return_tensors='list')

    batch_size = 8
  
    train_dataloader = DataLoader(lm_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(lm_datasets['valid'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(lm_datasets['test'], batch_size=batch_size, collate_fn=data_collator)

    model = cmbert.CMBertForMaskedLM(_model_config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    # num_epochs = 3
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
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    test_eval = evaluation(
        model=model,
        dataloader=test_dataloader,
    )

    result = {
        'dataset_config': dataset_config.__dict__,
        'training_arguments': training_arguments.__dict__,
        'model_config': _model_config.__dict__,
        'train_loss': train_loss,
        'valid_eval': valid_eval,
        'test_eval': test_eval,
    }

    save_result(result, results_path)
    model_checkpoint_name = model_config.encoder_checkpoint.split('/')[-1] + '_' + results_path.split('/')[-1][8:24]

    full_path = training_arguments.save_model_dest + '/' + model_checkpoint_name
    if training_arguments.save_best_model:
        best_model.save_pretrained(
            save_directory=full_path,
            state_dict=best_model.state_dict(),
        )

    with open('model_params.txt', 'w+') as fd:
        for param in best_model.parameters():
            fd.write(str(param))

if __name__ =='__main__':
    ds = 'CMUMOSI'
    text_features = SDK_DS[ds]['RAWTEXT']['featuresname']
    audio_features = SDK_DS[ds]['COVAREP']['featuresname']
    visual_features = SDK_DS[ds]['FACET42']['featuresname']
    labels = SDK_DS[ds]['LABELS']['featuresname']

    dataset_config = cmbert.CmuDatasetConfig(
        sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
        dataset = ds,
        text_features = text_features,
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels,
        preprocess = False,
        load_preprocessed = True,
    )

    _model_config = cmbert.CMBertConfig(
        encoder_checkpoint=model_config.encoder_checkpoint,
        modality_att_dropout_prob=model_config.modality_att_dropout_prob,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        hidden_size=model_config.hidden_size,
        audio_feat_size=SDK_DS[ds]['COVAREP']['feat_size'],
        visual_feat_size=SDK_DS[ds]['FACET42']['feat_size'],
        projection_size=model_config.projection_size,
        num_labels=model_config.num_labels,
        best_model_metric=model_config.best_model_metric,
    )

    training_arguments = TrainingArgs(
        batch_size = model_config.batch_size,
        num_epochs = model_config.num_epochs,
        criterion = model_config.criterion,
        optimizer = model_config.optimizer,
        lr = model_config.lr,
        scheduler_type = model_config.scheduler_type,
        warmup_steps_ratio = model_config.warmup_steps_ratio,
        save_best_model = True, # 
        save_model_dest = model_config.save_model_dest   
    )

    results_path = 'experiments/results_' + datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S') + '.jsonl'

    main(dataset_config=dataset_config, _model_config=_model_config, training_arguments=training_arguments, results_path=results_path)
    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # path = 'models/distilbert-base-uncased_05May2024_215644'
    # best_model = cmbert.CMBertForMaskedLM.from_pretrained(path).to(device)


    # model2 = cmbert.CMBertForSequenceClassification.from_pretrained(path).to(device)

    
    # with open('model_params3.txt', 'w+') as fd:
    #     for param in model2.parameters():
    #         fd.write(str(param))