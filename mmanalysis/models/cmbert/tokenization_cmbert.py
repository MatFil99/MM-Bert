from typing import Dict


from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
import numpy as np

# class MultimodalTokenizer(PreTrainedTokenizer):
class CMBertTokenizer():

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
                try:
                    audio[i] = audio[i][word_ids] # adjust audio to word tokens (if word was splitted, then we need to align audio)
                except:
                    # print(f'len {len(audio[i])}')
                    # print(f'word_ids {word_ids}')
                    
                    # 1
                    # word_ids = [min(id, len(audio[i])-1) for id in word_ids]

                    # 2
                    tokens = tokenized.tokens(i)
                    tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']]

                    token_words = self.get_token_words(tokens)

                    my_tokens = text[i].split()

                    words_map = self.get_words_map(token_words, my_tokens)
                    word_ids = self.get_new_word_ids(word_ids, words_map)
                    
                    audio[i] = audio[i][word_ids]
                    
                new_audio_i = np.pad(audio[i], ((1,max_lenght - len(audio[i]) - 1), (0,0)), constant_values=0)
                audio_padded.append(new_audio_i)
                
            if visual is not None:
                try:
                    visual[i] = visual[i][word_ids] # adjust visual - same as above
                except:
                    # print(f'len {len(visual[i])}')
                    # print(f'word_ids {word_ids}')
                    
                    # 1
                    word_ids = [min(id, len(visual[i])-1) for id in word_ids]
                    
                    # 2
                    tokens = tokenized.tokens(i)
                    tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']]

                    token_words = self.get_token_words(tokens)

                    my_tokens = text[i].split()

                    words_map = self.get_words_map(token_words, my_tokens)
                    word_ids = self.get_new_word_ids(word_ids, words_map)
                    
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
    

    def get_token_words(self, tokens):
        token_words = []
        word = tokens[0]
        for ti in range(len(tokens)):
            if ti+1 < len(tokens) \
                    and '##' in tokens[ti+1] \
                    and '##' not in tokens[ti]:
                word = tokens[ti]
            elif '##' in tokens[ti] \
                    and ti+1 < len(tokens) \
                    and '##' in tokens[ti+1]:
                word += tokens[ti].replace('##', '')
            elif '##' in tokens[ti] \
                    and ti+1 < len(tokens) \
                    and '##' not in tokens[ti+1]:
                word += tokens[ti].replace('##', '')
                token_words.append(word)
                word = "" if ti+1>=len(tokens) else tokens[ti+1]
            else:
                token_words.append(word)
                word = "" if ti+1>=len(tokens) else tokens[ti+1]

        return token_words

    def get_words_map(self, token_words, my_tokens):
        j = 0
        ti = 0

        words_map = {}
        while ti < len(token_words):
            if token_words[ti] == my_tokens[j]:
                words_map[ti] = j
                j+=1
                ti+=1
            else:
                pref_word = ''
                while ti < len(token_words) \
                        and pref_word + token_words[ti] in my_tokens[j]:
                    pref_word = pref_word + token_words[ti]
                    words_map[ti] = j
                    ti+=1
                j+=1

        return words_map
                
    def get_new_word_ids(self, word_ids, words_map):
        return [words_map[id] for id in word_ids]