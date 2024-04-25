from typing import Dict
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np

class MultimodalTokenizer():
    # class MultimodalTokenizer(PreTrainedTokenizer):

    def __init__(self, checkpoint):
        # we need to initialize text_tokenizer before calling super.__init__
        self.text_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # super(PreTrainedTokenizer, self).__init__()

    def get_vocab(self) -> Dict[str, int]:
        return self.text_tokenizer.get_vocab()
    
    def __call__(self, text, audio, visual, padding=True, truncation=True, return_tensors='pt'):
        tokenized = self.text_tokenizer(text, padding=padding, truncation=truncation, return_tensors=return_tensors)
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

        tokenized['audio_ids'] = np.array(audio_padded)
        tokenized['visual_ids'] = np.array(visual_padded)

        return tokenized
    