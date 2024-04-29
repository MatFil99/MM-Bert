
import cmbert
# import multimodal_tokenizer

if __name__ == '__main__':
    print("train.py")

    # dataset configuration
    dataset_config = cmbert.DataConfig()
    
    # model configuration
    checkpoint = 'google-bert/bert-base-uncased'
    model_config = cmbert.CMBertConfig()
    num_labels = 2


    # create dataset, model, tokenizer
    ds = cmbert.CmuDataset(dataset_config)
    model = cmbert.CMBertForSequenceClassification(config=model_config, num_labels=num_labels)
    tokenizer = cmbert.MultimodalTokenizer(checkpoint=checkpoint)

    # prepare data
    features, labels = ds.computational_sequences_2_array()
    text = ds.words_2_sentences()

    # select sample data
    n = 10
    text = text[:n]
    audio_feat = features['audio_feat'][:n]
    visual_feat = features['visual_feat'][:n]
    tokenized = tokenizer(text=text, audio=audio_feat, visual=visual_feat, padding=True, truncation=True)

    # # use all data
    # tokenized = tokenizer(text=text, audio=features['audio_feat'], visual=features['visual_feat'], padding=True, truncation=True)

    # remove visual features
    del tokenized['visual_data']

    logits, text_att, fusion_att = model(**tokenized)
    print(f'logits: {logits}')
    print(f'text_att: {text_att.shape}')
    print(f'fusion_att: {fusion_att.shape}')
    