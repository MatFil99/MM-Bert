import os

class CmuDatasetConfig():

    def __init__(self,
                 sdkpath = r'/home/filip/studies/CS_WUT/master_thesis/CMU-MultimodalSDK',
                 dataset = 'cmumosi',
                 text_features = 'CMU_MOSI_TimestampedWords',
                 audio_features = 'CMU_MOSI_COVAREP',
                 visual_features = 'CMU_MOSI_Visual_Facet_42',
                 labels = 'CMU_MOSI_Opinion_Labels',
                 preprocess = True,
                 load_preprocessed = False,
                 ):
        super(CmuDatasetConfig, self).__init__()

        self.sdkpath = sdkpath
        self.dataset = dataset.lower() #

        if load_preprocessed:
            self.ds_path = os.path.join(sdkpath, self.dataset, 'aligned')
        else:
            self.ds_path = os.path.join(sdkpath, self.dataset)

        self.feature_names = {}
        self.feature_names['text_feat'] = text_features
        if audio_features:
            self.feature_names['audio_feat'] = audio_features
        if visual_features:
            self.feature_names['visual_feat'] = visual_features

        self.labels = labels
        self.preprocess = preprocess
        self.load_preprocessed = load_preprocessed
