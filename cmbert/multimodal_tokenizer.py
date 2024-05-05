from typing import Dict


from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
import numpy as np

# class MultimodalTokenizer(PreTrainedTokenizer):
class MultimodalTokenizer():

    def __init__(self, checkpoint):
        # we need to initialize text_tokenizer before calling super.__init__
        self.text_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.mask_token_id = self.text_tokenizer.mask_token_id
        self.mask_token = self.text_tokenizer.mask_token
        
        # super(PreTrainedTokenizer, self).__init__()

    def get_vocab(self) -> Dict[str, int]:
        return self.text_tokenizer.get_vocab()  

    def __call__(self, text, audio, visual, padding=True, truncation=False, return_tensors='pt', device='cpu'):
        if return_tensors is None:
            tokenized = self.text_tokenizer(text, padding=padding, truncation=truncation)
        else:
            tokenized = self.text_tokenizer(text, padding=padding, truncation=truncation, return_tensors=return_tensors).to(device)
            max_lenght = tokenized.input_ids[0].shape[0]

        audio_padded = []
        visual_padded = []
        for i in range(len(tokenized['input_ids'])):
            max_lenght = len(tokenized.input_ids[i])
            word_ids = [word_id for word_id in tokenized.word_ids(i) if word_id is not None]
            
            # pad audio, visual data - two reserved for START and END tokens
            if audio is not None:
                audio[i] = audio[i][word_ids] # adjust audio to word tokens (if word was splitted, then we need to align audio)
                new_audio_i = np.pad(audio[i], ((1,max_lenght - len(audio[i]) - 1), (0,0)), constant_values=0)
                audio_padded.append(new_audio_i)
                
            if visual is not None:
                visual[i] = visual[i][word_ids] # adjust visual - same as above
                new_visual_i = np.pad(visual[i], ((1,max_lenght - len(visual[i]) - 1), (0,0)), constant_values=0)
                visual_padded.append(new_visual_i)

        if return_tensors is not None:
            tokenized['audio_data'] = np.array(audio_padded)
            tokenized['visual_data'] = np.array(visual_padded)
            if return_tensors == 'pt':
                tokenized['audio_data'] = torch.tensor(tokenized['audio_data']).to(torch.float32)
                tokenized['visual_data'] = torch.tensor(tokenized['visual_data']).to(torch.float32)
        elif audio is not None or visual is not None:
            if audio is not None:
                tokenized['audio_data'] = audio_padded
            if visual is not None:
                tokenized['visual_data'] = visual_padded
            
        return tokenized
    


    # def pad(self, *pad_args, **pad_kwargs):
    #     return self.text_tokenizer.pad(*pad_args, **pad_kwargs)
    
    # def get_special_tokens_mask(self, val, already_has_special_tokens=True):
    #     return self.text_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=already_has_special_tokens)
    
    # def convert_tokens_to_ids(self, mask_token):
    #     return self.text_tokenizer.convert_tokens_to_ids(mask_token)
    
    # def __len__(self):
    #     return len(self.text_tokenizer)