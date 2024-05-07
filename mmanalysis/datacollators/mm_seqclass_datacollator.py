import torch

class MMSeqClassDataCollator():

    def __init__(self, tokenizer, num_labels=2 ,device=torch.device("cpu")) -> None:
        self.tokenizer = tokenizer
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