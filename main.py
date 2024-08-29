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
    CmuDatasetConfig,
    CmuDataset,
    standard_folds
)


def get_model_config_class(model_name):
    if model_name == 'cmbert':
        return CMBertConfig
    elif model_name == 'mmbert':
        return MMBertConfig

def main_single_run():
    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    print(args)

    if args.ds is not None:
        base_config['dataset'] = args.ds
        ds = args.ds.upper()
    else: 
        ds = base_config['dataset'].upper()

    if 'class' in args.task:
        base_config['num_labels'] = int(args.task[5:])
    elif args.task == 'reg':
        base_config['num_labels'] = 1
    else:
        base_config['num_labels'] = None
    
    dataset, audio_feat_size, visual_feat_size = prepare_dataset(ds)
    train_ds, valid_ds, test_ds = train.prepare_data_splits(dataset, base_config['num_labels'])

    base_config_arr = {key: value for key, value in base_config.items() if type(value) != list}
    base_config_arr['save_model_dest'] = 'models/' + args.task
    run_config, _ = get_singlerun_configuration(**base_config_arr)

    (
        text_feat,
        model_name,
        encoder_checkpoint,
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

    # print(ModelConfigClass)

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
    
    
    results_path = 'experiments/' + args.task + '/results_' + model_name + '_' + datetime_start + '.jsonl'
    train.main(
        model_name=model_name,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        dataset_config=dataset.config,
        model_config=model_config,
        training_arguments=training_arguments,
        results_path=results_path,
        )

def prepare_dataset(ds):
    audio_features = None
    audio_feat_size = None
    visual_features = None
    visual_feat_size = None

    audio_feat = None if args.audio_feat is None else args.audio_feat
    visual_feat = None if args.visual_feat is None else args.visual_feat

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
        text_features = SDK_DS[ds][base_config['text_features']]['featuresname'],
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels,
        preprocess = args.preprocess,
        load_preprocessed = args.load_preprocessed, # by default load preprocessed data
    )
    dataset = CmuDataset(dataset_config)
    if args.dsdeploy:
        dataset.deploy()

    return dataset, audio_feat_size, visual_feat_size


def main_multirun():
    """
    """
    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    print(args)

    if args.ds is not None:
        base_config['dataset'] = args.ds
        ds = args.ds.upper()
    else: 
        ds = base_config['dataset'].upper()

    if 'class' in args.task:
        base_config['num_labels'] = int(args.task[5:])
    elif args.task == 'reg':
        base_config['num_labels'] = 1
    else:
        base_config['num_labels'] = None

    dataset, audio_feat_size, visual_feat_size = prepare_dataset(ds)
    train_ds, valid_ds, test_ds = train.prepare_data_splits(dataset, base_config['num_labels'])

    base_config_arr = {key: [value] for key, value in base_config.items() if type(value) != list}

    for name, config in multirun_configuration.items():
        param_config = copy.deepcopy(base_config_arr) # get base configuration 
        param_config[name] = config # assign values of tested parameter
        param_config['save_model_dest'] = ['models/' + args.task] 

        print(f'before config: {param_config}')

        # if args.ds:
        #     param_config['datasets'] = [args.ds]
        runs_config, _ = get_multirun_configuration(**param_config)

        print("Param {} - Number of runs configurations: {}, ds: {}, model {}".format(name, len(runs_config), param_config['dataset'], param_config['model_name']))

        i = 1
        for run_config in runs_config:
            print(f'config_run: {i} / {len(runs_config)}')
            print(run_config)
            i+=1

            (
                text_feat,
                model_name,
                encoder_checkpoint,
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

            train.main(
                model_name=model_name,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                dataset_config=dataset.config,
                model_config=model_config,
                training_arguments=training_arguments,
                results_path=results_path,
                )

def main_contrain_run():
    models_path = 'models/mlm/'
    experiments_path = 'experiments/mlm'

    datetime_start = datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S')

    if args.ds is not None:
        base_config['dataset'] = args.ds
        ds = args.ds.upper()
    else: 
        ds = base_config['dataset'].upper()

    if 'class' in args.task:
        base_config['num_labels'] = int(args.task[5:])
    elif args.task == 'reg':
        base_config['num_labels'] = 1
    else:
        base_config['num_labels'] = None

    dataset, audio_feat_size, visual_feat_size = prepare_dataset(ds)
    train_ds, valid_ds, test_ds = train.prepare_data_splits(dataset, base_config['num_labels'])

    base_config_arr = {key: [value] for key, value in base_config.items() if type(value) != list}

    models_checkpoints = sorted([dir for dir in os.listdir(models_path) if 'distilbert' in dir], reverse=True)

    print(f'models_checkpoints: {len(models_checkpoints)}')

    for checkpoint in models_checkpoints:
        print(checkpoint)
        checkpoint =  models_path + checkpoint
        run_cfg = get_pretrained_model_run_config(checkpoint, experiments_path)

        for name, config in multirun_configuration.items():
            param_config = copy.deepcopy(base_config_arr) # get base configuration 
            param_config[name] = config # assign values of tested parameter
            param_config['save_model_dest'] = ['models/' + args.task] 

            runs_config, _ = get_multirun_configuration(**param_config)
            print("Param {} - Number of runs configurations: {}, ds: {}, model {}".format(name, len(runs_config), param_config['dataset'], param_config['model_name']))

            i = 1
            for run_config in runs_config:
                print(f'config_run: {i} / {len(runs_config)}')
                print(run_config)
                i+=1

                (
                    _,
                    _,
                    encoder_checkpoint,
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
                projection_size = run_cfg['model_config']['projection_size']
                
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

                train.main(
                    model_name=model_name,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    dataset_config=dataset.config,
                    model_config=model_config,
                    training_arguments=training_arguments,
                    results_path=results_path,
                    )

def main_evaluate():
    task = args.task
    ds = args.ds.upper()
    text_feat = args.text_feat
    audio_feat = args.audio_feat
    visual_feat = args.visual_feat
    models_path = 'models/' + args.task + '/'

    models_checkpoints = sorted([dir for dir in os.listdir(models_path) if 'bert' in dir], reverse=True)

    print(f'models_checkpoints: {len(models_checkpoints)}')

    if text_feat is not None:
        text_features = SDK_DS[ds][text_feat]['featuresname']
    if audio_feat is not None:
        audio_features = SDK_DS[ds][audio_feat]['featuresname']
    if visual_feat is not None:
        visual_features = SDK_DS[ds][visual_feat]['featuresname']
    labels_features = SDK_DS[ds]['LABELS']['featuresname']
                    
    dataset_config = CmuDatasetConfig(
        sdkpath = SDK_DS['SDK_PATH'],
        dataset = ds,
        text_features = text_features,
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels_features, 
        preprocess = args.preprocess,
        load_preprocessed = args.loadpreprocessed,
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
            # contrain_configuration,
            multirun_configuration,
            base_config,
            train,
        )
    elif args.task == 'mlm':
        # import mlm training configuration
        from mmanalysis.training.mlm import (
            # continuedtrain_config,
            multirun_configuration,
            base_config,
            train,
        )
    elif args.task == 'reg':
        from mmanalysis.training.regression import (
            # contrain_configuration,
            multirun_configuration,
            base_config,
            train,
        )
    
    from mmanalysis.training.get_config import (
        get_singlerun_configuration,
        get_multirun_configuration
    )

    if args.eval:
        main_evaluate()
    elif args.contrain:
        main_contrain_run()
    elif args.multirun:
        main_multirun()
    else:
        main_single_run()
        
