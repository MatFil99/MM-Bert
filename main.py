import json
import os
import argparse
from datetime import datetime
import copy

# import train_class as train
from mmanalysis.training.training_arguments import TrainingArgs

# import mmanalysis.models.cmbert
from mmanalysis.models.cmbert import CMBertConfig
from mmanalysis.models.mmbert import MMBertConfig

from mmanalysis.datasets.cmu import (
    SDK_DS,
    CmuDatasetConfig
)

def get_model_config_class(model_name):
    if model_name == 'cmbert':
        return CMBertConfig
    elif model_name == 'mmbert':
        return MMBertConfig

def main_single_run(args):
    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    ds = args.ds.upper()
    text_features = SDK_DS[ds][args.text_feat]['featuresname']
    audio_features = None
    audio_feat_size = None
    visual_features = None
    visual_feat_size = None
    labels = SDK_DS[ds]['LABELS']['featuresname']

    if args.audio_feat is not None:
        audio_features = SDK_DS[ds][args.audio_feat]['featuresname']
        audio_feat_size = SDK_DS[ds][args.audio_feat]['feat_size']
    if args.visual_feat is not None:
        visual_features = SDK_DS[ds][args.visual_feat]['featuresname']
        visual_feat_size = SDK_DS[ds][args.visual_feat]['feat_size']

    (
    model_name,
    encoder_checkpoint,
    # load_pretrained,
    hidden_dropout_prob,
    modality_att_dropout_prob,
    freeze_params_layers,
    hidden_size,
    projection_size,
    num_labels,
    batch_size,
    num_epochs,
    patience,
    chunk_size,
    wwm_probability,
    criterion,
    optimizer,
    layer_specific_optimization,
    lr,
    scheduler_type,
    warmup_steps_ratio,
    best_model_metric,
    save_best_model,
    save_model_dest
    ) = get_singlerun_configuration(**singlerun_configuration)

    dataset_config = CmuDatasetConfig(
        sdkpath = SDK_DS['SDK_PATH'],
        dataset = ds,
        text_features = text_features,
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels,
        preprocess = args.preprocess,
        load_preprocessed = args.load_preprocessed,
    )

    training_arguments = TrainingArgs(
        batch_size = batch_size,
        num_epochs = num_epochs,
        patience = patience,
        chunk_size = chunk_size,
        wwm_probability = wwm_probability,
        criterion = criterion,
        optimizer = optimizer,
        layer_specific_optimization = layer_specific_optimization,
        lr = lr,
        scheduler_type = scheduler_type,
        warmup_steps_ratio = warmup_steps_ratio,
        save_best_model = save_best_model,
        save_model_dest = save_model_dest
    )

    # CMBertConfig / MMBertConfig / ...
    ModelConfigClass = get_model_config_class(model_name=model_name)

    model_config = ModelConfigClass(
        encoder_checkpoint=encoder_checkpoint,
        modality_att_dropout_prob=modality_att_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        freeze_params_layers=freeze_params_layers,
        hidden_size=hidden_size,
        audio_feat_size=audio_feat_size,
        visual_feat_size=visual_feat_size,
        projection_size=projection_size,
        num_labels=num_labels,
        best_model_metric=best_model_metric,
    )
    
    # results_path = 'experiments/results_' + model_name + '_' + datetime_start + '.jsonl'
    results_path = 'experiments/' + args.task + '/results_' + model_name + '_' + datetime_start + '.jsonl'
    train.main(
        model_name=model_name,
        dataset_config=dataset_config,
        model_config=model_config,
        training_arguments=training_arguments,
        results_path=results_path,
        dsdeploy=args.dsdeploy,
        )


def main_multirun(args):
    """
    """
    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')
    
    param_configs = {
        # 'hidden_dropout_prob': [0.1, 0.2],
        # 'modality_att_dropout_prob': [0.1, 0.2, 0.3],
        # 'projection_size': [30],
        # 'freeze_params_layers': [5, 6],
        
        'hidden_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
        'modality_att_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
        'projection_size': [30, 60, 120, 240, 480, 768],
        'freeze_params_layers': [0, 3, 4, 5, 6],
    }

    if args.task == 'class2':
        num_class = [2]
    elif args.task == 'class7':
        num_class = [7]

    for name, config in param_configs.items():
        multirun_param_config = copy.deepcopy(multirun_configuration)
        multirun_param_config[name] = config
        multirun_param_config['save_model_dest'] = ['models/' + args.task]
        if args.task in ['class2', 'class7']:
            multirun_param_config['num_labels'] = num_class
        if args.ds:
            multirun_param_config['datasets'] = [args.ds]
        datasets, runs_config = get_multirun_configuration(**multirun_param_config)

        print("Param {} - Number of runs configurations: {}, ds: {}, model {}".format(name, len(runs_config), multirun_param_config['datasets'], multirun_param_config['model_names']))

        for dataset in datasets:
            ds = dataset.upper()

            i = 1
            for run_config in runs_config:
                print(f'config_run: {i} / {len(runs_config)}')
                i+=1
                audio_features = None
                audio_feat_size = None
                visual_features = None
                visual_feat_size = None

                (
                    text_feat,
                    audio_feat,
                    visual_feat,
                    model_name,
                    encoder_checkpoint,
                    # load_pretrained,
                    hidden_dropout_prob,
                    modality_att_dropout_prob,
                    freeze_params_layers,
                    hidden_size,
                    projection_size,
                    num_labels,
                    batch_size,
                    num_epochs,
                    patience,
                    chunk_size,
                    wwm_probability,
                    criterion,
                    optimizer,
                    layer_specific_optimization,
                    lr,
                    scheduler_type,
                    warmup_steps_ratio,
                    best_model_metric,
                    save_best_model,
                    save_model_dest
                ) = run_config


                visual_features = None
                visual_feat_size = None
                labels = SDK_DS[ds]['LABELS']['featuresname']
                if audio_feat is not None:
                    audio_features = SDK_DS[ds][audio_feat]['featuresname']
                    audio_feat_size = SDK_DS[ds][audio_feat]['feat_size']
                if visual_feat is not None:
                    visual_features = SDK_DS[ds][visual_feat]['featuresname']
                    visual_feat_size = SDK_DS[ds][visual_feat]['feat_size']

                dataset_config = CmuDatasetConfig(
                    sdkpath = SDK_DS['SDK_PATH'],
                    dataset = ds,
                    text_features = SDK_DS[ds][text_feat]['featuresname'],
                    audio_features = audio_features,
                    visual_features = visual_features,
                    labels = labels,
                    preprocess = False,
                    load_preprocessed = True, # by default load preprocessed data
                )

                training_arguments = TrainingArgs(
                    batch_size = batch_size,
                    num_epochs = num_epochs,
                    patience = patience,
                    chunk_size = chunk_size,
                    wwm_probability = wwm_probability,
                    criterion = criterion,
                    optimizer = optimizer,
                    layer_specific_optimization = layer_specific_optimization,
                    lr = lr,
                    scheduler_type = scheduler_type,
                    warmup_steps_ratio = warmup_steps_ratio,
                    save_best_model = save_best_model,
                    save_model_dest = save_model_dest
                )

                ModelConfigClass = get_model_config_class(model_name=model_name)
                
                model_config = ModelConfigClass(
                    encoder_checkpoint = encoder_checkpoint,
                    modality_att_dropout_prob = modality_att_dropout_prob,
                    hidden_dropout_prob = hidden_dropout_prob,
                    hidden_size = hidden_size,
                    freeze_params_layers=freeze_params_layers,
                    audio_feat_size = audio_feat_size,
                    visual_feat_size = visual_feat_size,
                    projection_size = projection_size,
                    num_labels = num_labels,
                    best_model_metric = best_model_metric,
                )
                
                results_path = 'experiments/' + args.task + '/results_' + model_name + '_' + datetime_start + '.jsonl'

                # print(dataset_config.__dict__)
                # print(f"model_config.freeze_params_layers {model_config.freeze_params_layers}")

                train.main(
                    model_name=model_name,
                    dataset_config=dataset_config,
                    model_config=model_config,
                    training_arguments=training_arguments,
                    results_path=results_path,
                    dsdeploy=args.dsdeploy,
                    )

def main_contrain_run(args):
    models_path = 'models/mlm/'
    experiments_path = 'experiments/mlm'
    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')
    
    datasets, runs_config = get_multirun_configuration(**contrain_configuration)

    models_checkpoints = sorted([dir for dir in os.listdir(models_path) if 'bert' in dir], reverse=True)

    print(f'run_config: {len(runs_config)}')
    print(f'models_checkpoints: {len(models_checkpoints)}')

    for dataset in datasets:
        ds = dataset.upper()
        
        for checkpoint in models_checkpoints:
            print(checkpoint)
            checkpoint =  models_path + checkpoint
            run_cfg = get_pretrained_model_run_config(checkpoint, experiments_path)
            
            i = 1
            for run_config in runs_config:
                print(f'config_run: {i} / {len(runs_config)}')
                
                i+=1
                audio_feat = None
                visual_feat = None

                (
                    text_feat,
                    audio_feat,
                    visual_feat,
                    _,
                    encoder_checkpoint,
                    # load_pretrained,
                    hidden_dropout_prob,
                    modality_att_dropout_prob,
                    freeze_params_layers,
                    hidden_size,
                    _,
                    num_labels,
                    batch_size,
                    num_epochs,
                    patience,
                    chunk_size,
                    wwm_probability,
                    criterion,
                    optimizer,
                    layer_specific_optimization,
                    lr,
                    scheduler_type,
                    warmup_steps_ratio,
                    best_model_metric,
                    save_best_model,
                    save_model_dest
                ) = run_config
                
                model_name = run_cfg['model_name']

                if text_feat is not None:
                    text_features = SDK_DS[ds][text_feat]['featuresname']
                if audio_feat is not None:
                    audio_features = SDK_DS[ds][audio_feat]['featuresname']
                    audio_feat_size = SDK_DS[ds][audio_feat]['feat_size']
                if visual_feat is not None:
                    visual_features = SDK_DS[ds][visual_feat]['featuresname']
                    visual_feat_size = SDK_DS[ds][visual_feat]['feat_size']
                labels_features = SDK_DS[ds]['LABELS']['featuresname']
                # if 'audio_feat' in run_cfg['dataset_config']['feature_names']:
                #     audio_feat = run_cfg['dataset_config']['feature_names']['audio_feat']
                # if 'visual_feat' in run_cfg['dataset_config']['feature_names']:
                #     visual_feat = run_cfg['dataset_config']['feature_names']['visual_feat']
                projection_size = run_cfg['model_config']['projection_size']
                # print(run_cfg['dataset_config']['feature_names'])

                dataset_config = CmuDatasetConfig(
                    sdkpath = SDK_DS['SDK_PATH'],
                    dataset = ds,
                    text_features = text_features, # run_cfg['dataset_config']['feature_names']['text_feat'],
                    audio_features = audio_features,
                    visual_features = visual_features,
                    labels = labels_features, # run_cfg['dataset_config']['labels'],
                    preprocess = run_cfg['dataset_config']['preprocess'],
                    load_preprocessed = True, # by default load preprocessed data
                )
                
                training_arguments = TrainingArgs(
                    batch_size = batch_size,
                    num_epochs = num_epochs,
                    patience = patience,
                    chunk_size = chunk_size,
                    wwm_probability = wwm_probability,
                    criterion = criterion,
                    optimizer = optimizer,
                    layer_specific_optimization = layer_specific_optimization,
                    lr = lr,
                    scheduler_type = scheduler_type,
                    warmup_steps_ratio = warmup_steps_ratio,
                    save_best_model = save_best_model,
                    save_model_dest = save_model_dest
                )

                ModelConfigClass = get_model_config_class(model_name=model_name)
                
                model_config = ModelConfigClass(
                    encoder_checkpoint = encoder_checkpoint,
                    modality_att_dropout_prob = modality_att_dropout_prob,
                    hidden_dropout_prob = hidden_dropout_prob,
                    hidden_size = hidden_size,
                    freeze_params_layers = freeze_params_layers,
                    audio_feat_size = run_cfg['model_config']['audio_feat_size'],
                    visual_feat_size = run_cfg['model_config']['visual_feat_size'],
                    projection_size = projection_size,
                    num_labels = num_labels,
                    best_model_metric = best_model_metric,
                )

                results_path = 'experiments/' + args.task + str(num_labels) + '/results_' + model_name + '_pretrained_' + datetime_start + '.jsonl'

                # print(f'dataset_config {dataset_config.__dict__}')
                # print(f'training_arguments {training_arguments.__dict__}')
                # print(f'model_config {model_config}')
                # print(f'results_path {results_path}')
                train.main(
                    model_name=model_name,
                    dataset_config=dataset_config,
                    model_config=model_config,
                    training_arguments=training_arguments,
                    results_path=results_path,
                    pretrained_checkpoint=checkpoint,
                    dsdeploy=args.dsdeploy,
                )
            
            # break


def main_evaluate(args):
    task = args.task
    ds = args.ds.upper()
    text_feat = args.text_feat
    audio_feat = args.audio_feat
    visual_feat = args.visual_feat
    models_path = 'models/' + args.task + '/'
    experiments_path = 'experiments/' + args.task

    models_checkpoints = sorted([dir for dir in os.listdir(models_path) if 'bert' in dir], reverse=True)

    print(f'models_checkpoints: {len(models_checkpoints)}')

    # run_cfg = get_pretrained_model_run_config(checkpoint, experiments_path)
    if text_feat is not None:
        text_features = SDK_DS[ds][text_feat]['featuresname']
    if audio_feat is not None:
        audio_features = SDK_DS[ds][audio_feat]['featuresname']
        audio_feat_size = SDK_DS[ds][audio_feat]['feat_size']
    if visual_feat is not None:
        visual_features = SDK_DS[ds][visual_feat]['featuresname']
        visual_feat_size = SDK_DS[ds][visual_feat]['feat_size']
    labels_features = SDK_DS[ds]['LABELS']['featuresname']
                    
    dataset_config = CmuDatasetConfig(
        sdkpath = SDK_DS['SDK_PATH'],
        dataset = ds,
        text_features = text_features, # run_cfg['dataset_config']['feature_names']['text_feat'],
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels_features, # run_cfg['dataset_config']['labels'],
        preprocess = False,
        load_preprocessed = True, # by default load preprocessed data
    )

    train.main_evaluate(checkpoints=models_checkpoints, dataset_config=dataset_config, task=task)

    
    

def get_pretrained_model_run_config(checkpoint, exp_path):
    configs = None
    checkpoint_id = checkpoint[-16:]
    files = [f for f in os.listdir(exp_path) if 'jsonl' in f]
    for file in files:
        with open(os.path.join(exp_path,file), 'r') as fd:
            configs = fd.readlines()
        for c in configs:
            cfg = json.loads(c)
            if cfg['datetime_run'] == checkpoint_id:
                return cfg
    
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='cmbert',
        description=
        """
        Program runs training model and return test evaluation.
        """,
        epilog='Model for multimodal sentiment classification.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

# arguments for single run
    parser.add_argument('-ds', metavar='dataset', type=str,
                        choices=['cmumosi', 'cmumosei', 'pom'],
                        help='dataset for training and testing model'
                        )

    parser.add_argument('-m', metavar='model', type=str,
                        choices=['cmbert', 'mmbert'],
                        help='model for training'
                        )

    parser.add_argument('-task', dest='task',
                        choices=['class', 'reg', 'mlm', 'class2', 'class7'], default='class',
                        help='train model with task'
                        )

    parser.add_argument('-t', dest='text_feat', metavar='text features', type=str,
                        choices=['RAWTEXT'], default='RAWTEXT',
                        help='text features used for training and testing model'
                        )

    parser.add_argument('-a', dest='audio_feat', metavar='audio features', type=str,
                        choices=['COVAREP', 'OPENSMILE'], # OPENSMILE only for cmumosi
                        help='audio features used for training and testing model'
                        )
    
    parser.add_argument('-v', dest='visual_feat', metavar='visual features', type=str,
                        choices=['FACET42', 'OPENFACE2'],
                        help='visual features used for training and testing model'
                        )

    parser.add_argument('--preprocess', dest='preprocess',
                        action='store_true',
                        help='do preprocessing dataset, it takse long time - better reuse saved preprocessed data'
                        )
    
    parser.add_argument('--loadpreprocessed', dest='load_preprocessed',
                        action='store_true',
                        help='load preprocessed data - saved preprocessed data required'
                        )
    
    parser.add_argument('--dsdeploy', dest='dsdeploy',
                        action='store_true',
                        help='save preprocessed dataset'
                        )

    parser.add_argument('--multirun', dest='multirun',
                        action='store_true',
                        help='run trainings for all configurations defined in runs_config.py'
                        )
    
    parser.add_argument('--contrain', dest='contrain',
                        action='store_true',
                        help='run trainings on pretrained models (continue training)'
                        )
    
    parser.add_argument('--eval', dest='eval',
                        action='store_true',
                        help='run evaluation on training dataset for all models in directory'
                        )

    args = parser.parse_args()

    if 'class' in args.task:
        from mmanalysis.training.classification import (
            contrain_configuration,
            multirun_configuration,
            singlerun_configuration,
            train,
        )
    elif args.task == 'mlm':
        # import mlm training configuration
        from mmanalysis.training.mlm import (
            # continuedtrain_config,
            multirun_configuration,
            singlerun_configuration,
            train,
        )
    elif args.task == 'reg':
        from mmanalysis.training.regression import (
            contrain_configuration,
            multirun_configuration,
            singlerun_configuration,
            train,
        )
    
    from mmanalysis.training.get_config import (
        get_singlerun_configuration,
        get_multirun_configuration
    )

    if args.eval:
        main_evaluate(args)
    elif args.contrain:
        main_contrain_run(args)
    elif args.multirun:
        main_multirun(args)
    else:
        main_single_run(args)
        
