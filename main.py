import argparse
import itertools
from datetime import datetime

# from cmbert import SDK_DS

# import train_class as train
from mmanalysis.training.training_arguments import TrainingArgs

# import mmanalysis.models.cmbert
from mmanalysis.models.cmbert import CMBertConfig
from mmanalysis.models.mmbert import MMBertConfig

from mmanalysis.datasets.cmu import (
    SDK_DS,
    CmuDatasetConfig
)

# parameters for single run
# import model_config



# parameters for multirun
# import runs_config

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

    (
    encoder_checkpoint,
    hidden_dropout_prob,
    modality_att_dropout_prob,
    hidden_size,
    projection_size,
    num_labels,
    batch_size,
    num_epochs,
    chunk_size,
    criterion,
    optimizer,
    lr,
    scheduler_type,
    warmup_steps_ratio,
    best_model_metric,
    save_best_model,
    save_model_dest
    ) = get_singlerun_configuration()

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
        chunk_size=chunk_size,
        criterion = criterion,
        optimizer = optimizer,
        lr = lr,
        scheduler_type = scheduler_type,
        warmup_steps_ratio = warmup_steps_ratio,
        save_best_model = save_best_model,
        save_model_dest = save_model_dest
    )

    _model_config = CMBertConfig(
    # _model_config = MMBertConfig(
        encoder_checkpoint=encoder_checkpoint,
        modality_att_dropout_prob=modality_att_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        hidden_size=hidden_size,
        audio_feat_size=audio_feat_size,
        visual_feat_size=visual_feat_size,
        projection_size=projection_size,
        num_labels=num_labels,
        best_model_metric=best_model_metric,
    )

    train.main(
        dataset_config=dataset_config,
        model_config=_model_config,
        training_arguments=training_arguments,
        results_path=results_path,
        dsdeploy=args.dsdeploy,
        )


def main_multirun(args):
    """
    """
    results_path = 'experiments/results_' + datetime.strftime(datetime.now(), format='%d%b%Y_%H%M%S') + '.jsonl'
    
    
    datasets, runs_config = get_runs_configuration()

    for dataset in datasets:
        ds = dataset.upper()

        for run_config in runs_config:
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
                chunk_size,
                num_epochs,
                criterion,
                optimizer,
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
                chunk_size=chunk_size,
                criterion = criterion,
                optimizer = optimizer,
                lr = lr,
                scheduler_type = scheduler_type,
                warmup_steps_ratio = warmup_steps_ratio,
                save_best_model = save_best_model,
                save_model_dest = save_model_dest
            )

            _model_config = CMBertConfig(
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
                dsdeploy=args.dsdeploy,
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
    parser.add_argument('-ds', metavar='dataset', type=str,
                        choices=['cmumosi', 'cmumosei', 'pom'],
                        help='dataset for training and testing model'
                        )

    parser.add_argument('-m', metavar='model', type=str,
                        choices=['cmbert', 'mmbert'],
                        help='model for training'
                        )

    parser.add_argument('-task', dest='task',
                        choices=['class', 'reg', 'mlm'], default='class',
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
    

    args = parser.parse_args()

    if args.task == 'class':
        from mmanalysis.training.classification import (
            get_singlerun_configuration,
            get_runs_configuration,
            train,
        )
    elif args.task == 'mlm':
        # import mlm training configuration
        from mmanalysis.training.mlm import (
            get_singlerun_configuration,
            get_runs_configuration,
            train,
        )
    elif args.task == 'reg':
        from mmanalysis.training.regression import (
            get_singlerun_configuration,
            get_runs_configuration,
            train,
        )
    

    if not args.multirun:
        main_single_run(args)
    else:
        main_multirun(args)

