# import sys
# sys.path.append('..\\cmbert')

# import cmbert
# # # import 
# # import cmu_dataset

import cmbert

def download_datasets_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    ds.download_datasets()


def load_data_t1():
    config = cmbert.DataConfig()
    ds = cmbert.CmuDataset(config)
    assert ds.dataset.keys() is not None

    # print(ds.dataset.keys())
    # print(ds.dataset['CMU_MOSI_Visual_Facet_42'])

    all_segments = []
    for feat in ds.dataset.keys():
        all_segments.extend([segid for segid in ds.dataset[feat].keys()])

    print(all_segments)


def remove_unmatched_segments_t1():
    config = cmbert.DataConfig()
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