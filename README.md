# MM-Bert
CM-Bert implementation - model for multimodal sentiment analysis, extended with the visual modality
MM-Bert implementation - modified CM-Bert model with different attention mechanism


### Prerequisities
Model works with CMU-MultimodalSDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/main <br>

Clone repository to some directory (remember path)
git clone git@github.com:CMU-MultiComp-Lab/CMU-MultimodalSDK.git <br>

Create cmumosi, cmumosei and pom directories in CMU-MultimodalSDK catalog (required to store downloaded datasets and run MM-Bert program)

### MM-Bert installation
#### Clone repo
```
git clone git@github.com:MatFil99/MM-Bert.git
```

#### Create and activate venv
```
cd MM-Bert
python -m venv venv <br>
venv\Scripts\activate 
```

#### Create directories with experiments and models
```
python init_dirs.py
```

#### Install required packages
```
pip install -r requirements.txt
```

#### Provide path to SDK
Edit files mmanalysis/datasets/cmu/configuration_cmu.py, manalysis/datasets/cmu/sdk_datasets.py
Provide full path to installed CMU-MultimodalSDK - change value of SDK_PATH variable (template shown below).
```
mmanalysis/datasets/cmu/configuration_cmu.py
sdkpath = r'D:\path-to-sdk\CMU-MultimodalSDK'

manalysis/datasets/cmu/sdk_datasets.py
sdk_path = r'D:\path-to-sdk\CMU-MultimodalSDK'
```

### How to use
#### Arguments:
* <b>-ds</b> - dataset, choices: ['cmumosi', 'cmumosei', 'pom']
* <b>-task</b> - task, choices: ['reg', 'mlm', 'class2', 'class7']
* <b>-t</b> - text features, choices: ['RAWTEXT'] - Word Timestamps
* <b>-a</b> - audio features, choices: ['COVAREP', 'OPENSMILE'] - details in mmanalysis/datasets/cmu/sdk_datasets.py
* <b>-v</b> - visual features, choices: ['FACET42'] - details in mmanalysis/datasets/cmu/sdk_datasets.py
* <b>--preprocess</b> - if preprocess should be executed (it takes much time), best way is to preprocess and deploy for first run of dataset and then use 
--loadpreprocessed option
* <b>--loadpreprocessed</b> - if data are already preprocessed and the step could be skipped
* <b>--dsdeploy</b> - whether save preprocessed data or not
* <b>--multirun</b> - execute multiple parameters tests (many training with different parameters specified in base_config.py and multirun_config.py)
* <b>--contrain</b> - use core of every model located in models/mlm folder to continue training for specified task
* <b>--eval</b> - run just evaluation


#### Save preprocessed data for first run (recommended)
```
# cmumosi
python .\main.py -ds cmumosi -a COVAREP -v FACET42 -task class2 --preprocess --dsdeploy

# cmumosei
python .\main.py -ds cmumosei -a COVAREP -v FACET42 -task class2 --preprocess --dsdeploy

# pom
python .\main.py -ds pom -a COVAREP -v FACET42 -task class2 --preprocess --dsdeploy
```

#### Run single training
1. Set parameters in mmanalysis/training/[task]/base_config.py
2. Run program <br>
Example below - training on cmumosi dataset, using text, audio (COVAREP), visual (FACET42) features. Task was set to classification (with 2 classes), loadpreprocessed causes using saved aligned data (instead of preparing it from original) <br>
```
python .\main.py -ds cmumosi -a COVAREP -v FACET42 -task class2 --loadpreprocessed
```

#### Run multirun training - test many values of parameter(s)
1. Set basic parameters in file mmanalysis/training/[task]/base_config.py
2. Define values for parameters that will be tested in file mmanalysis/training/[task]/multirun_config.py
Example: <br>
```
contrain_configuration = {
    'hidden_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
    'freeze_params_layers': [0, 3, 4, 5, 6],
}
```
Configuration above will test two parameters. It will set basic configuration and then execute training for each value of hidden_dropout_prob. Next reset to basic configuration and act the same for freeze_params_layers.
3. Run program with --multirun flag
Example:
```
python .\main.py -ds cmumosi -a COVAREP -task class2 --loadpreprocessed --multirun
```

#### Continue training - use pretrained models
1. Store pretrained models in models/mlm directory (program will execute training for every model inside this directory)
2. Set parameters that will be tested in mmanalysis/training/[task]/multirun_config.py . Be careful to do not test parameters that are specific for model architecture (these are enforced by chosen pretrained model)
3. Run program with --contrain flag
Example:
```
python .\main.py -ds cmumosi -a COVAREP -v FACET42 -task class2 --loadpreprocessed --contrain
```


