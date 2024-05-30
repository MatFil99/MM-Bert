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
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds_mosi
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI import cmu_mosei_std_folds as standard_folds_mosei
from mmsdk.mmdatasdk.dataset.standard_datasets.POM import pom_std_folds as standard_folds_pom

standard_folds = {
    'cmumosi': standard_folds_mosi,
    'cmumosei': standard_folds_mosei,
    'pom': standard_folds_pom,
}

from .cmu_dataset import CmuDataset