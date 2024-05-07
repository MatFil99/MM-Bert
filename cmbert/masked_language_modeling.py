

from datasets.dataset_dict import DatasetDict
from datasets import Dataset

from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds
import cmbert




def prepare_data_splits(ds, num_labels):
    # ds.labels_2_class(num_classes=num_labels)

    folds = {
        'train': standard_folds.standard_train_fold,
        'valid': standard_folds.standard_valid_fold,
        'test': standard_folds.standard_test_fold,
        }
   
    ds.train_test_valid_split(folds=folds)
    ds.replace_inf_and_nan_values(value=0)
    ds.computational_sequences_2_array()
    ds.words_2_sentences()

    train_ds = cmbert.CmuDataset.from_dataset(ds, fold='train')
    test_ds = cmbert.CmuDataset.from_dataset(ds, fold='test')
    valid_ds = cmbert.CmuDataset.from_dataset(ds, fold='valid')
        
    return train_ds, valid_ds, test_ds

def cmudataset_to_datasetdict(train_ds, valid_ds, test_ds):
    return DatasetDict({
        'train': Dataset.from_dict(train_ds.dataset),
        'valid': Dataset.from_dict(valid_ds.dataset),
        'test': Dataset.from_dict(test_ds.dataset),
        # 'train': Dataset.from_dict(train_ds[:225]),
        # 'valid': Dataset.from_dict(valid_ds[:225]),
        # 'test': Dataset.from_dict(test_ds[:225]),
    })
    
# def 

