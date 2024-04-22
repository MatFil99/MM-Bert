import sys
import os
import numpy as np
from .utils import *

SDKPATH = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK'
sys.path.append(SDKPATH)
# import mmsdk
from mmsdk import mmdatasdk as md


class DataConfig():

    def __init__(self,
                 sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
                 dataset = 'cmumosi',
                 ds_path = None,
                 text_features = 'CMU_MOSI_TimestampedWords',
                 audio_features = 'CMU_MOSI_COVAREP',
                 visual_features = 'CMU_MOSI_Visual_Facet_42',
                 labels = 'CMU_MOSI_Opinion_Labels'
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



class CmuDataset():

    def __init__(self, config):
        super(CmuDataset, self).__init__()

        self.dsname = config.dataset
        self.ds_path = config.ds_path
        self.feature_names = config.feature_names
        self.labels_name = config.labels

        self.dataset, self.labels_ds = self.load_data()

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


    def remove_unmatched_segments(self):
        """
        """
        label_feat = self.labels_name

        segments = []
        for feat in self.dataset.keys():
            segments.extend(list(self.dataset[feat].keys()))
        segments.extend(self.labels_ds.keys())
        all_segments = set(segments)
        missing_segments = set()

        for segment in all_segments:
            for feat in self.dataset.keys():
                if segment not in self.dataset[feat].keys():
                    missing_segments.add(segment)
            if segment not in self.labels_ds[label_feat].keys():
                missing_segments.add(segment)
        
        for segment in missing_segments:
            for feat in self.dataset.keys():
                if segment in self.dataset[feat].keys():
                    # to remove elements from dataset we need use compuational_sequences 
                    del self.dataset.computational_sequences[feat].data[segment]

            if segment in self.labels_ds[label_feat].keys():
                del self.labels_ds.computational_sequences[label_feat].data[segment]
            

    def align_features(self, mode = 'text_feat', collapse_functions = [avg]):
        self.dataset.align(self.feature_names[mode], collapse_functions = collapse_functions)

    def align_labels(self):
        self.dataset.computational_sequences[self.labels_name] = self.labels_ds.computational_sequences[self.labels_name]
        self.dataset.align(self.labels_name)