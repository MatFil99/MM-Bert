# provide path to sdk downloaded from repository: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/main
sdk_path = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK'

class SDKDatasets(dict):

    def __init__(self):
        super(SDKDatasets, self).__init__()
        self['SDK_PATH'] = sdk_path
        self['CMUMOSI'] = {}
        self['CMUMOSEI'] = {}
        self.initialize_datasets_config()

    def initialize_datasets_config(self):
        # CMUMOSI
        self['CMUMOSI']['RAWTEXT'] = {
            'featuresname': 'CMU_MOSI_TimestampedWords',
            'feat_size': 1,
            'modality': 'rawtext'
        }
        self['CMUMOSI']['COVAREP'] = {
            'featuresname': 'CMU_MOSI_COVAREP',
            'feat_size': 74,
            'modality': 'audio'
        }
        self['CMUMOSI']['OPENSMILE'] = {
            'featuresname': 'CMU_MOSI_openSMILE_IS09',
            'feat_size': 384,
            'modality': 'audio'
        }
        self['CMUMOSI']['FACET42'] = {
            'featuresname': 'CMU_MOSI_Visual_Facet_42',
            'feat_size': 35,
            'modality': 'visual'
        }
        self['CMUMOSI']['LABELS'] = {
            'featuresname': 'CMU_MOSI_Opinion_Labels',
            'feat_size': 1,
            'modality': 'label'
        }

        # CMUMOSEI
        self['CMUMOSEI']['RAWTEXT'] = {
            'featuresname': 'CMU_MOSEI_TimestampedWords',
            'feat_size': 1,
            'modality': 'rawtext'
        }
        self['CMUMOSEI']['COVAREP'] = {
            'featuresname': 'CMU_MOSEI_COVAREP',
            'feat_size': 74,
            'modality': 'audio'
        }
        self['CMUMOSEI']['FACET42'] = {
            'featuresname': 'CMU_MOSEI_VisualFacet42',
            'feat_size': 35,
            'modality': 'visual'
        }
        self['CMUMOSEI']['OPENFACE2'] = {
            'featuresname': 'CMU_MOSEI_VisualOpenFace2',
            'feat_size': 709, # after selection just useful features
            'modality': 'visual'
        }
        self['CMUMOSEI']['LABELS'] = {
            'featuresname': 'CMU_MOSEI_Labels',
            'feat_size': 1,
            'modality': 'label'
        }

# create sdk configuration variable
SDK_DS = SDKDatasets()