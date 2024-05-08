import json

import numpy as np
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam, RMSprop, Adadelta


def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features 


def bword_vector_2_sentence(bwords, filter = [b'sp'], encoding='utf-8'):
    return ' '.join([bw.decode(encoding) for bw in bwords if bw not in filter])


def group_texts(examples, chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
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


def set_optimizer_custom_parameters(
          model, 
          weight_decay=0.01, 
          lr=0.01, 
          no_decay_layers=['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight'],
          new_decay_layers=['modality_fusion']
          ):
    # Prepare optimizer
    # if ('audio_feat' in dataset_config.feature_names 
    #     or 'visual_feat' in dataset_config.feature_names):
    param_optimizer = list(model.named_parameters())
    no_decay = no_decay_layers
    new_decay = new_decay_layers

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(np in n for np in new_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay )and any(np in n for np in new_decay)],'lr':lr}
    ]

    return optimizer_grouped_parameters

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
   

