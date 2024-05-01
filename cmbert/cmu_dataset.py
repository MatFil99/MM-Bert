import sys
import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

# from utils import *
from .utils import *


SDKPATH = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK'
sys.path.append(SDKPATH)
# import mmsdk
from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk.configurations.metadataconfigs import featuresetMetadataTemplate

class DataConfig():

    def __init__(self,
                 sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
                 dataset = 'cmumosi',
                 ds_path = None,
                 text_features = 'CMU_MOSI_TimestampedWords',
                 audio_features = 'CMU_MOSI_COVAREP',
                 visual_features = 'CMU_MOSI_Visual_Facet_42',
                 labels = 'CMU_MOSI_Opinion_Labels',
                 load_preprocessed = False,
                 remove_pause = True # remove b'sp' words from raw text features
                 ):
        super(DataConfig, self).__init__()

        self.sdkpath = sdkpath
        self.dataset = dataset

        if ds_path:
            self.ds_path = ds_path
        else:
            self.ds_path = os.path.join(sdkpath, dataset)

        self.feature_names = {
            'text_feat' : text_features,
            'audio_feat' : audio_features,
            'visual_feat' : visual_features,
        }
        self.labels = labels
        self.load_preprocessed = load_preprocessed

class CmuDataset(Dataset):

    def __init__(self, config, ds=None, preprocess=True):
        super(CmuDataset, self).__init__()

        self.dsname = config.dataset
        self.ds_path = config.ds_path
        self.feature_names = config.feature_names
        self.labels_name = config.labels
        self.preprocessed = config.load_preprocessed
        
        if ds:
            self.dataset = ds
        else:
            self.dataset, self.labels_ds = self.load_data()
        
        if config.load_preprocessed:
            self._standardize_loaded_data()

        if preprocess and not ds:
            # for testing
            self._cut_to_n_videos(10)

            self.align_features(mode='text_feat')
            self.remove_special_text_tokens(keep_aligned=True)
            self.append_labels_to_dataset() # append labels to dataset and then align data to labels
            self.align_to_labels() #
            # self.labels_2_class(num_classes=2)
            self.preprocessed = preprocess
        
        # self.labels = self.dataset[self.labels_name]
        
    @classmethod
    def from_dataset(cls, cmudataset, fold):
        config = DataConfig(
                sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
                dataset = cmudataset.dsname,
                ds_path = cmudataset.ds_path,
                text_features = cmudataset.feature_names['text_feat'],
                audio_features = cmudataset.feature_names['audio_feat'],
                visual_features = cmudataset.feature_names['visual_feat'],
                labels = cmudataset.labels_name,
        ) 

        if fold:
            dataset = cmudataset.dataset[fold]
        else:
            dataset = cmudataset.dataset

        return cls(config, dataset, preprocess=False)

    def __getitem__(self, index) -> dict:
        if 'train' in self.dataset:
            ds = self.dataset['train']
        else:
            ds = self.dataset

        data = {}
        for feat in ds.keys():
            data[feat] = ds[feat][index]
        
        return data
        # return super().__getitem__(index)

    def __len__(self) -> int:
        """
        """
        key = list(self.dataset.keys())[0]
        return len(self.dataset[key])

    def _standardize_loaded_data(self):
        """
        """
        dictdata = {feat: defaultdict(dict) for feat in self.dataset.keys()}
        # intervals = defaultdict(dict)
        
        for feat in self.dataset.keys():
            comseq = self.dataset[feat]
            for segid in comseq.keys():
                dictdata[feat][segid]['features'] = comseq[segid]['features'][:]
                dictdata[feat][segid]['intervals'] = comseq[segid]['intervals'][:]

        for feat in dictdata.keys():
            self.dataset.computational_sequences[feat] = dictdata[feat]
            # self.dataset.computational_sequences[feat] = features[feat]


    def _cut_to_n_videos(self, n=10):
        """
        """
        vid_ids = set(list(self.labels_ds[self.labels_name].keys())[:n])
        all_vid_ids = self._get_all_segments()
        vid_to_remove = all_vid_ids - vid_ids
        
        for feat in self.feature_names.values():
            for id in vid_to_remove:
                if id in self.dataset[feat].keys():
                    del self.dataset[feat].data[id]
                #  for sig in self.dataset[feat].keys():

    def deploy(self, destination=None, filenames=None):
        for key in self.dataset.keys():
            self.initialize_missing_metadata(self.dataset.computational_sequences[key])

        if destination is None:
            destination = self.ds_path + '\\aligned'
        if filenames is None:
            filenames = {feat: feat for feat in self.dataset.keys()}
        self.dataset.deploy(destination, filenames)

    def initialize_missing_metadata(self, comseq):
        missings=[x for (x,y) in zip(featuresetMetadataTemplate,[metadata in comseq.metadata.keys() for metadata in featuresetMetadataTemplate]) if y is False]
        for m in missings:
            comseq.metadata[m] = ''

    def load_data(self):
        recipe_features = {
            feat: self.ds_path + '\\' + feat + '.csd' for feat in self.feature_names.values() if feat is not None
        }
        recipe_labels = {
            self.labels_name : self.ds_path + '\\' + self.labels_name + '.csd'
        }

        try:
            ds = md.mmdataset(recipe_features)
            labels = md.mmdataset(recipe_labels)
        except RuntimeError as e:
            print(e)
            ds = None
            labels = None

        return ds, labels


    def download_datasets(self):
        # create folders for storing the data
        if not os.path.exists(self.ds_path):
            os.mkdir(self.ds_path)

        if self.dsname == 'cmumosi':
            DATASET = md.cmu_mosi
        elif self.dsname == 'cmumosei':
            DATASET = md.cmu_mosei
        elif self.dsname == 'pom':
            DATASET = md.pom
        
        try:
            md.mmdataset(DATASET.highlevel, self.ds_path)
        except RuntimeError:
            print('High-level features have been downloaded previously.')
        
        try:
            md.mmdataset(DATASET.raw, self.ds_path)
        except RuntimeError:
            print('Raw data have been downloaded previously.')
            
        try:
            md.mmdataset(DATASET.labels, self.ds_path)
        except RuntimeError:
            print('Labels have been downloaded previously.')

    def _get_all_segments(self):
        segments = []
        for feat in self.dataset.keys():
            segments.extend(list(self.dataset[feat].keys()))
        if not self.preprocessed:
            segments.extend(self.labels_ds.keys())
        return set(segments)

    def remove_unmatched_segments(self):
        """
        """
        label_feat = self.labels_name

        all_segments = self._get_all_segments()
        missing_segments = set()

        for segment in all_segments:
            for feat in self.dataset.keys():
                if segment not in self.dataset[feat].keys():
                    missing_segments.add(segment)
            if not self.preprocessed and segment not in self.labels_ds[label_feat].keys():
                missing_segments.add(segment)
        
        for segment in missing_segments:
            for feat in self.dataset.keys():
                if segment in self.dataset[feat].keys():
                    # to remove elements from dataset we need use compuational_sequences 
                    del self.dataset.computational_sequences[feat].data[segment]

            if segment in self.labels_ds[label_feat].keys():
                del self.labels_ds.computational_sequences[label_feat].data[segment]
            
    def remove_special_text_tokens(self, tokens = [b'sp'], keep_aligned=True):
        """
        keep_aligned - parameter that defines if remove corresponding segments from other's modality data
        """
        text_comseq = self.dataset.computational_sequences[self.feature_names['text_feat']]
        
        segids = [id for id in text_comseq.keys() if text_comseq[id]['features'][0,0] in tokens]
        for sid in segids:
            del text_comseq.data[sid]

            if keep_aligned:
                for feat in self.feature_names.values():
                    comseq_feat = self.dataset.computational_sequences[feat]
                    if sid in comseq_feat.data:
                        del comseq_feat.data[sid]  

    def align_features(self, mode = 'text_feat', collapse_functions = [avg]):
        self.dataset.align(self.feature_names[mode], collapse_functions = collapse_functions)

    def align_to_labels(self):
        # firstly append labels to dataset
        if self.labels_name in self.dataset.keys():
            self.dataset.align(self.labels_name)
        else:
            raise RuntimeError # if labels are not present in the dataset then throw exception

    def word_vectors_2_sentences(self):
        text_features = {}

        for segid in self.dataset[self.feature_names['text_feat']].keys():
            word_vec = self.dataset[self.feature_names['text_feat']][segid]['features'][:].squeeze(1)
            sentence = bword_vector_2_sentence(word_vec)
            text_features[segid] = sentence

        return text_features

    def append_labels_to_dataset(self):
        self.dataset.computational_sequences[self.labels_name] = self.labels_ds.computational_sequences[self.labels_name]

    def computational_sequences_2_array(self, fold=None):
        """
        """

        if fold is None and self.labels_name in self.dataset.keys():
            ds = self.dataset
            features = self._computational_sequences_2_array(ds)
        if fold in self.dataset.keys():
            ds = self.dataset[fold]
            features = self._computational_sequences_2_array(ds)
        else:
            features = {}
            for _fold in self.dataset.keys():
                ds = self.dataset[_fold]
                features[_fold] = self._computational_sequences_2_array(ds)

        for key in self.dataset.keys():
            self.dataset.computational_sequences[key] = features[key]

    def _computational_sequences_2_array(self, ds):
        """
        """

        # if fold in self.dataset.keys():
        #     ds = self.dataset[fold]
        # else:
        #     ds = self.dataset
        
        if self.preprocessed:
            segments = ds[self.labels_name].keys()
        else:
            segments = self.labels_ds[self.labels_name].keys()

        features = {feat: [] for feat in list(self.feature_names.keys()) + ['labels']}
        # labels = []
        
        for segid in segments:
            for feat_key, feat_name in self.feature_names.items():
                feat_values = ds[feat_name][segid]['features']
                features[feat_key].append(feat_values)

            if self.preprocessed:
                features['labels'].extend(ds[self.labels_name][segid]['features'])
                # labels.append(ds[self.labels_name][segid]['features'])
            else:
                features['labels'].extend(self.labels_ds[self.labels_name][segid]['features'])
                # labels.append(self.labels_ds[self.labels_name][segid]['features'])

        return features #, np.array(labels).squeeze(1)

    # def words_2_sentences(self, fold=None):
    #     """
    #     """
    #     if fold in self.dataset.keys():
    #         text_features = self.dataset[fold][self.feature_names['text_feat']]
    #     else:
    #         text_features = self.dataset[self.feature_names['text_feat']]

    #     sentences = []
    #     # if not text_features:
    #     #     text_features = self.dataset.computational_sequences[self.feature_names['text_feat']]

    #     for segid in text_features.keys():
    #         words = text_features[segid]['features'][:].squeeze(1)
    #         sentences.append(bword_vector_2_sentence(words))

    #     return sentences

    def words_2_sentences(self, fold=None):
        """
        """
        # if fold in self.dataset.keys():
        #     ds = self.dataset[fold]
        #     sentences = self._words_2_sentences(ds)
        # elif self.labels_name in self.dataset.keys(): 
        #     ds = self.dataset
        #     sentences = self._words_2_sentences(ds)
        # else:
        for fold in self.dataset.keys():
            ds = self.dataset[fold]
            sentences = self._words_2_sentences(ds)

        

    def _words_2_sentences(self, ds):
        sentences = []
        text_features = ds['text_feat']

        for words in text_features:
            # words = text_features[segid][:].squeeze(1)
            sentences.append(bword_vector_2_sentence(words.squeeze(1)))

        ds['text_feat'] = sentences

        return sentences

    def _filter_observations(self, vids_filter):
        ds = self.dataset

        if self.preprocessed:
            segments = ds[self.labels_name].keys()
            labels_ds = self.dataset[self.labels_name]
        else:
            segments = self._get_all_segments()
            labels_ds = self.labels_ds[self.labels_name]

        filtered = defaultdict(dict)
        for segid in segments:
            vid = segid.split('[')[0]
            if vid in vids_filter:
                for feat in self.feature_names.values():
                    if segid in ds[feat].keys():
                        filtered[feat][segid] = ds[feat][segid]
                
                if segid in labels_ds.keys():
                    filtered[self.labels_name][segid] = labels_ds[segid]

        return filtered
                

    def train_test_valid_split(self, folds: dict):
        """
        """
        foldsplits = {fold: defaultdict(dict) for fold in folds}

        for fold, video_ids in folds.items():
            foldsplits[fold] = self._filter_observations(video_ids)

            self.dataset.computational_sequences[fold] = foldsplits[fold]

        for feat in self.feature_names.values():
            del self.dataset.computational_sequences[feat]
        if self.preprocessed:
            del self.dataset.computational_sequences[self.labels_name]

        return foldsplits
    

    def labels_2_class(self, num_classes=2, inplace=True):
        """
        """
        if num_classes == 2:
            if self.labels_name in self.dataset.keys():
                ds = self.dataset
                self._labels_2_class(ds, num_classes=num_classes)
            else:
                for fold, ds in self.dataset.keys():
                    self._labels_2_class(ds, num_classes=num_classes)
        elif num_classes == 7:
            if self.labels_name in self.dataset.keys():
                ds = self.dataset
                self._labels_2_class(ds, num_classes=num_classes)
            else:
                for fold, ds in self.dataset.keys():
                    self._labels_2_class(ds, num_classes=num_classes)


    def _labels_2_class(self, dset, num_classes=2, inplace=True):
        """
        """
        ds = dset.computational_sequences[self.labels_name]
        classes = defaultdict(dict)
        
        for segid in ds.keys():
            if num_classes == 2:
                newlabels = np.array([self._label_2_two_class(val) for val in ds[segid]['features'][:].squeeze(1)])
            elif num_classes == 7:
                newlabels = np.array([self._label_2_seven_classes(val) for val in ds[segid]['features'][:].squeeze(1)])
            
            classes[segid]['features'] = newlabels
            classes[segid]['intervals'] = ds[segid]['intervals'][:]

        if inplace:
            dset.computational_sequences[self.labels_name] = classes
        else:
            dset.computational_sequences['labels'] = classes

        
    def _label_2_two_class(self, a):
        if a <= 0:
            res = 0
        else:
            res = 1

        return res


    def _label_2_seven_classes(self, a):
        if a < -2:
            res = 0
        if -2 <= a and a < -1:
            res = 1
        if -1 <= a and a < 0:
            res = 2
        if 0 <= a and a <= 0:
            res = 3
        if 0 < a and a <= 1:
            res = 4
        if 1 < a and a <= 2:
            res = 5
        if a > 2:
            res = 6
        
        return res
    