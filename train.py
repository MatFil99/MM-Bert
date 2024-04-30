from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import get_scheduler, AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator
from tqdm.auto import tqdm

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

def train(model, tokenizer, ds):
    seed = 7
    transformers.set_seed(seed)
    torch.manual_seed(seed)

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
        'train': ['03bSnISJMiM', '0h-zjBukYpk', '1DmNV9C1hbY', '1iG0909rllw', '2WGyTLYerpo'], # standard_folds.standard_train_fold,
        'valid': ['2iD-tVS8NPw', '5W7Z1C_fDaE', '6Egk_28TtTM'], # standard_folds.standard_valid_fold,
        'test': ['6_0THN4chvY', '73jzhE8R1TQ']# standard_folds.standard_test_fold
        }
    ds.train_test_valid_split(folds=folds)
    
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = cmbert.CmuDataset.from_dataset(ds, fold='train')
    test_ds = cmbert.CmuDataset.from_dataset(ds, fold='test')
    valid_ds = cmbert.CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds

def train(model, train_dataloader, valid_dataloader, num_epochs, optimizer, lr_scheduler, criterion):
    # num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    
    for epoch in range(num_epochs):
        model.train()
        cum_loss = 0.0
        for batch in train_dataloader:
            del batch['visual_data']

            logits, _, _ = model(**batch)

            loss = criterion(logits, batch['labels'])
            loss.backward()
            cum_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        print('==================================')
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print(f'Training loss: {cum_loss}')
        # print('Validation: ', evaluation)


if __name__ == '__main__':
    print("train.py")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # dataset configuration
    dataset_config = cmbert.DataConfig()
    
    # model configuration
    checkpoint = 'google-bert/bert-base-uncased'
    model_config = cmbert.CMBertConfig()
    num_labels = 2


    # create dataset, model, tokenizer
    ds = cmbert.CmuDataset(dataset_config)
    model = cmbert.CMBertForSequenceClassification(config=model_config, num_labels=num_labels)
    # tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)

    # prepare data
    train_ds, valid_ds, test_ds = prepare_data_splits(ds=ds)
    

    # # select sample data
    # n = 10
    # # text = text[:n]
    # # audio_feat = features['audio_feat'][:n]
    # # visual_feat = features['visual_feat'][:n]

    # text = data['train']['features']['text_feat'][:n]
    # audio_feat = data['train']['features']['audio_feat'][:n]
    # visual_feat = data['train']['features']['visual_feat'][:n]

    # tokenized = tokenizer(text=text, audio=audio_feat, visual=visual_feat, padding=True, truncation=True)

    # # # use all data
    # # tokenized = tokenizer(text=text, audio=features['audio_feat'], visual=features['visual_feat'], padding=True, truncation=True)

    # # remove visual features
    # del tokenized['visual_data']


    # 
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    criterion.to(device)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_collator = MultiModalDataCollator(checkpoint=checkpoint, device=device)

    train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_ds, batch_size=4, collate_fn=data_collator)

    num_epochs = 3
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
        
        # break
    # print(i)
    # print(sentences)


    train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
    )

    # text = tokenized['input_ids'].to(device)
    # tokenized['audio_data'].to(device)
    
    # print(tokenized['input_ids'].is_cuda)
    # print(tokenized['audio_data'].is_cuda)



    # data_collator = DefaultDataCollator()
    # # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # train_dataloader = DataLoader(
    #     tokenized, shuffle=True, batch_size=4, collate_fn=data_collator
    # )

    # print(train_dataloader)

    # for batch in train_dataloader:
    #     print(batch)

    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )

    # logits, text_att, fusion_att = model(**tokenized)
    # print(f'logits: {logits}')
    # print(f'text_att: {text_att.shape}')
    # print(f'fusion_att: {fusion_att.shape}')
    
    # print(f'labels: {data["valid"]["labels"][:2]}')