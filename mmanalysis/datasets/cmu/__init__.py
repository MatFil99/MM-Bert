import sys
UTILS_PATH = 'mmanalysis/utils'
sys.path.append(UTILS_PATH)


from .sdk_datasets import SDK_DS
from .configuration_cmu import CmuDatasetConfig

# import SDK tools
SDKPATH= SDK_DS['SDK_PATH']
sys.path.append(SDKPATH)
# import mmsdk
from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk.configurations.metadataconfigs import featuresetMetadataTemplate
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds



from .cmu_dataset import CmuDataset