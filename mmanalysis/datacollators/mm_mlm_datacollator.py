import collections


import numpy as np
import torch


class MMMLMDataCollator():

    def __init__(self, tokenizer, wwm_probability=0.2, return_tensors='list', device=torch.device('cpu')) -> None:
        self.tokenizer = tokenizer
        self.wwm_probability = wwm_probability
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
            mask = np.random.binomial(1, self.wwm_probability, (len(mapping),))
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
