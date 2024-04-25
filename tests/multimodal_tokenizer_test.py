import cmbert
# import cmbert.multimodal_tokenizer

def init_t1():
    checkpoint = 'google-bert/bert-base-uncased'
    tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)    

def tokenize_t1():
    config = cmbert.DataConfig()
    checkpoint = 'google-bert/bert-base-uncased'

    ds = cmbert.CmuDataset(config)

    # print(ds.keys())

    features, labels = ds.computational_sequences_2_array()

    tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)
    text = ds.words_2_sentences()

    print(f'text length: {len(text)}')
    print(f'text: {text[0]}')
    
    tokenized = tokenizer(text=text, audio=features['audio_feat'], visual=features['visual_feat'], padding=True, truncation=True)

    print(tokenized['input_ids'].shape)
    print(tokenized['audio_ids'].shape)
    print(tokenized['visual_ids'].shape)