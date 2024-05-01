from typing import Dict


from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
import numpy as np

# class MultimodalTokenizer():
class MultimodalTokenizer(PreTrainedTokenizer):

    def __init__(self, checkpoint):
        # we need to initialize text_tokenizer before calling super.__init__
        self.text_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        super(PreTrainedTokenizer, self).__init__()

    def get_vocab(self) -> Dict[str, int]:
        return self.text_tokenizer.get_vocab()
    
    # def to(self, device):
    #     self.text_tokenizer.to(device)

    # def __call__(self, text_feat, audio_feat, visual_feat, labels=None, padding=True, truncation=True, return_tensors='pt', device='cpu'):
    #     tokenized = self.text_tokenizer(text_feat, padding=padding, truncation=truncation, return_tensors=return_tensors).to(device)
    #     max_lenght = tokenized.input_ids[0].shape[0]

    #     audio_padded = []
    #     visual_padded = []
    #     for i in range(len(tokenized['input_ids'])):
    #         word_ids = [word_id for word_id in tokenized.word_ids(i) if word_id is not None]
    #         audio_feat[i] = audio_feat[i][word_ids] # adjust audio to word tokens (if word was splitted, then we need to align audio)
    #         visual_feat[i] = visual_feat[i][word_ids] # adjust visual - same as above

    #         # pad audio, visual data - two reserved for START and END tokens
    #         new_audio_i = np.pad(audio_feat[i], ((1,max_lenght - len(audio_feat[i]) - 1), (0,0)), constant_values=0)
    #         new_visual_i = np.pad(visual_feat[i], ((1,max_lenght - len(visual_feat[i]) - 1), (0,0)), constant_values=0)

    #         audio_padded.append(new_audio_i)
    #         visual_padded.append(new_visual_i)

    #     tokenized['audio_data'] = np.array(audio_padded)
    #     tokenized['visual_data'] = np.array(visual_padded)
    #     if return_tensors == 'pt':
    #         tokenized['audio_data'] = torch.tensor(tokenized['audio_data'])
    #         tokenized['visual_data'] = torch.tensor(tokenized['visual_data'])

    #     return tokenized
    

    def __call__(self, text, audio, visual, padding=True, truncation=True, return_tensors='pt', device='cpu'):
        tokenized = self.text_tokenizer(text, padding=padding, truncation=truncation, return_tensors=return_tensors).to(device)
        max_lenght = tokenized.input_ids[0].shape[0]

        audio_padded = []
        visual_padded = []
        for i in range(len(tokenized['input_ids'])):
            word_ids = [word_id for word_id in tokenized.word_ids(i) if word_id is not None]
            audio[i] = audio[i][word_ids] # adjust audio to word tokens (if word was splitted, then we need to align audio)
            visual[i] = visual[i][word_ids] # adjust visual - same as above

            # pad audio, visual data - two reserved for START and END tokens
            new_audio_i = np.pad(audio[i], ((1,max_lenght - len(audio[i]) - 1), (0,0)), constant_values=0)
            new_visual_i = np.pad(visual[i], ((1,max_lenght - len(visual[i]) - 1), (0,0)), constant_values=0)

            audio_padded.append(new_audio_i)
            visual_padded.append(new_visual_i)

        tokenized['audio_data'] = np.array(audio_padded)
        tokenized['visual_data'] = np.array(visual_padded)
        if return_tensors == 'pt':
            tokenized['audio_data'] = torch.tensor(tokenized['audio_data'])
            tokenized['visual_data'] = torch.tensor(tokenized['visual_data'])

        return tokenized
    