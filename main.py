import argparse
import itertools
from datetime import datetime

import cmbert
from cmbert import SDK_DS
import train

# parameters for single run
import model_config

# parameters for multirun
import runs_config

def main_single_run(args):
    results_path = 'experiments/results_' + datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S') + '.jsonl'
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

    dataset_config = cmbert.CmuDatasetConfig(
        sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
        dataset = ds,
        text_features = text_features,
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels,
        preprocess = args.preprocess,
        load_preprocessed = args.load_preprocessed,
    )

    training_arguments = train.TrainingArgs(
        batch_size = model_config.batch_size,
        num_epochs = model_config.num_epochs,
        criterion = model_config.criterion,
        optimizer = model_config.optimizer,
        lr = model_config.lr,
        scheduler_type = model_config.scheduler_type,
        num_warmup_steps = model_config.num_warmup_steps,
        save_best_model = model_config.save_best_model,
        save_model_dest = model_config.save_model_dest
    )

    _model_config = cmbert.CMBertConfig(
        encoder_checkpoint=model_config.encoder_checkpoint,
        modality_att_dropout_prob=model_config.modality_att_dropout_prob,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        hidden_size=model_config.hidden_size,
        audio_feat_size=audio_feat_size,
        visual_feat_size=visual_feat_size,
        projection_size=model_config.projection_size,
        num_labels=model_config.num_labels,
        best_model_metric=model_config.best_model_metric,
    )

    # for debug
    # for attr, value in dataset_config.__dict__.items():
    #     print(f'{attr}: {value}')

    # print()
    # for attr, value in training_arguments.__dict__.items():
    #     print(f'{attr}: {value}')
    
    # print()
    # for attr, value in _model_config.__dict__.items():
    #     print(f'{attr}: {value}')
    # for debug
    # print(_model_config.best_model_metric)

    train.main(
        dataset_config=dataset_config,
        model_config=_model_config,
        training_arguments=training_arguments,
        results_path=results_path,
        )

def permute_run_params():
    """
    """
    # datasets = runs_config.datasets # independently
    text_features = runs_config.text_features
    audio_features = runs_config.audio_features
    visual_features = runs_config.visual_features

    encoder_checkpoints = runs_config.encoder_checkpoints
    hidden_dropout_prob = runs_config.hidden_dropout_prob
    modality_att_dropout_prob = runs_config.modality_att_dropout_prob
    hidden_size = runs_config.hidden_size
    projection_size = runs_config.projection_size
    num_labels = runs_config.num_labels

    batch_size = runs_config.batch_size
    num_epochs = runs_config.num_epochs
    criterion = runs_config.criterion
    optimizer = runs_config.optimizer
    lr = runs_config.lr
    scheduler_type = runs_config.scheduler_type
    num_warmup_steps = runs_config.num_warmup_steps
    best_model_metric = runs_config.best_model_metric
    save_best_model = runs_config.save_best_model
    save_model_dest = runs_config.save_model_dest

    classification_runs = set(itertools.product(*[
        text_features,
        audio_features,
        visual_features,
        encoder_checkpoints,
        hidden_dropout_prob,
        modality_att_dropout_prob,
        hidden_size,
        projection_size,
        [labels for labels in num_labels if labels != 1], # num_labels=1 for regression
        batch_size,
        num_epochs,
        criterion['classification'],
        optimizer,
        lr,
        scheduler_type,
        num_warmup_steps,
        best_model_metric['classification'],
        save_best_model,
        save_model_dest
    ]))

    regression_runs = set(itertools.product(*[
        text_features,
        audio_features,
        visual_features,
        encoder_checkpoints,
        hidden_dropout_prob,
        modality_att_dropout_prob,
        hidden_size,
        projection_size,
        [labels for labels in num_labels if labels == 1], # num_labels=1 for regression
        batch_size,
        num_epochs,
        criterion['regression'],
        optimizer,
        lr,
        scheduler_type,
        num_warmup_steps,
        best_model_metric['regression'],
        save_best_model,
        save_model_dest
    ]))

    return classification_runs, regression_runs

def main_multirun(args):
    """
    """
    results_path = 'experiments/results_' + datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S') + '.jsonl'
    
    classification_runs, regression_runs = permute_run_params()
    all_runs = classification_runs.union(regression_runs)

    datasets = runs_config.datasets

    for dataset in datasets:
        ds = dataset.upper()

        for run_config in all_runs:
            audio_features = None
            audio_feat_size = None
            visual_features = None
            visual_feat_size = None

            (
                text_feat,
                audio_feat,
                visual_feat,
                encoder_checkpoint,
                hidden_dropout_prob,
                modality_att_dropout_prob,
                hidden_size,
                projection_size,
                num_labels,
                batch_size,
                num_epochs,
                criterion,
                optimizer,
                lr,
                scheduler_type,
                num_warmup_steps,
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

            dataset_config = cmbert.CmuDatasetConfig(
                sdkpath = SDK_DS['SDK_PATH'],
                dataset = ds,
                text_features = SDK_DS[ds][text_feat]['featuresname'],
                audio_features = audio_features,
                visual_features = visual_features,
                labels = labels,
                preprocess = args.preprocess,
                load_preprocessed = args.load_preprocessed,
            )

            training_arguments = train.TrainingArgs(
                batch_size = batch_size,
                num_epochs = num_epochs,
                criterion = criterion,
                optimizer = optimizer,
                lr = lr,
                scheduler_type = scheduler_type,
                num_warmup_steps = num_warmup_steps,
                save_best_model = save_best_model,
                save_model_dest = save_model_dest
            )

            _model_config = cmbert.CMBertConfig(
                encoder_checkpoint = encoder_checkpoint,
                modality_att_dropout_prob = modality_att_dropout_prob,
                hidden_dropout_prob = hidden_dropout_prob,
                hidden_size = hidden_size,
                audio_feat_size = audio_feat_size,
                visual_feat_size = visual_feat_size,
                projection_size = projection_size,
                num_labels = num_labels,
                best_model_metric = best_model_metric,
            )

            train.main(
                dataset_config=dataset_config,
                model_config=_model_config,
                training_arguments=training_arguments,
                results_path=results_path,
                )



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
    parser.add_argument('--ds', metavar='dataset', type=str,
                        choices=['cmumosi', 'cmumosei', 'pom'],
                        help='dataset for training and testing model'
                        )

    parser.add_argument('--t', dest='text_feat', metavar='text features', type=str,
                        choices=['RAWTEXT'], default='RAWTEXT',
                        help='text features used for training and testing model'
                        )

    parser.add_argument('--a', dest='audio_feat', metavar='audio features', type=str,
                        choices=['COVAREP'], 
                        help='audio features used for training and testing model'
                        )
    
    parser.add_argument('--v', dest='visual_feat', metavar='visual features', type=str,
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

# multiple runs
    parser.add_argument('--multirun', dest='multirun',
                        action='store_true',
                        help='if run trainings for all configurations defined in runs_config.py'
                        )
    


    args = parser.parse_args()

    if not args.multirun:
        main_single_run(args)
    else:
        main_multirun(args)

