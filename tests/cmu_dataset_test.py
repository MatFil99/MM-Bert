import sys

import cmbert

SDKPATH = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK'
sys.path.append(SDKPATH)
# import mmsdk
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds


def download_datasets_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    ds.download_datasets()


def load_data_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    assert ds.dataset.keys() is not None


def remove_unmatched_segments_t1():
    config = cmbert.DataConfig()
    # ds = cmbert.CmuDataset(config, preprocess=False)
    ds = cmbert.CmuDataset(config)

    text_feat = ds.feature_names['text_feat']
    print(f'Before removing: {len(ds.dataset[text_feat].keys())}')

    ds.remove_unmatched_segments()
    print(f'After removing: {len(ds.dataset[text_feat].keys())}')


def align_features_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    
    ds.align_features()


def align_labels_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    
    ds.align_labels()

    print(ds.dataset.keys())


def word_vectors_2_sentences_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)

    text_features = ds.word_vectors_2_sentences()

    print(text_features.keys())
    print(text_features['zhpQhgha_KU[33]'])


def remove_special_text_tokens_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)

    # print(ds.dataset[ds.labels_name].keys())
    ds.align_features(mode='text_feat')
    ds.remove_special_text_tokens(keep_aligned=True)
    ds.append_labels_to_dataset() # append labels to dataset and then align data to labels
    ds.align_to_labels() # 


def computational_sequences_2_array_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)

    features, labels = ds.computational_sequences_2_array()

    for name, feat in features.items():
        print(f'{name}: {len(feat)} {feat[0].shape}')

    print(f'labels.shape: {len(labels)} {labels[0].shape}')

def computational_sequences_2_array_t2():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)

    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold
        }
    
    ds.train_test_valid_split(folds)
    train_features, train_labels = ds.computational_sequences_2_array(fold='train')
    valid_features, valid_labels = ds.computational_sequences_2_array(fold='valid')
    test_features, test_labels = ds.computational_sequences_2_array(fold='test')

    print(f'train features: {train_features.keys()}, labels length: {len(train_labels)}')



def words_2_sentences_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)

    sentences = ds.words_2_sentences()

    print(sentences[0])
    print(sentences[1])
    print(sentences[2])


def train_test_valid_split_t1():
    config = cmbert.DataConfig(preprocess=True)
    ds = cmbert.CmuDataset(config)

    # md.dataset.standard_datasets.CMU_MOSI.
    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold
        }
    
    result = ds.train_test_valid_split(folds)

    print('Returned dictionary')
    print(result.keys())
    print(result['train'].keys())

    print()

    print('Dataset')
    print(ds.dataset.keys())
    print(ds.dataset['train'].keys())
    print(ds.dataset['train']['CMU_MOSI_COVAREP'].keys())
    print(ds.dataset['train']['CMU_MOSI_COVAREP']['W8NXH0Djyww[21]']['features'][:])
    



# to paste in prepare_data_splits
def dataset_consistency_after_manipulation(ds):
    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold,
        # 'train': ['03bSnISJMiM', '0h-zjBukYpk', '1DmNV9C1hbY', '1iG0909rllw', '2WGyTLYerpo'], # standard_folds.standard_train_fold,
        # 'valid': ['2iD-tVS8NPw', '5W7Z1C_fDaE', '6Egk_28TtTM'], # standard_folds.standard_valid_fold,
        # 'test': ['6_0THN4chvY', '73jzhE8R1TQ']# standard_folds.standard_test_fold
        }
    
    segments = list(ds.dataset['CMU_MOSI_TimestampedWords'].keys())
    # print(segments)

    idx = 15
    segid = segments[idx]
    # print(idx)

    print(ds.dataset['CMU_MOSI_TimestampedWords'][segid])
    print(ds.dataset['CMU_MOSI_Opinion_Labels'][segid])
    

    ds.train_test_valid_split(folds=folds)

    print(ds.dataset['train']['CMU_MOSI_TimestampedWords'][segid])
    print(ds.dataset['train']['CMU_MOSI_Opinion_Labels'][segid])
    
    ds.computational_sequences_2_array()

    print(ds.dataset['train']['text_feat'][idx])
    print(ds.dataset['train']['labels'][idx])

    # print(ds.dataset.keys())
    # print(ds.dataset['train'].keys())
    # print(ds.dataset['train']['text_feat'][0])
    
    ds.words_2_sentences()

    print(ds.dataset['train']['text_feat'][idx])
    print(ds.dataset['train']['labels'][idx])