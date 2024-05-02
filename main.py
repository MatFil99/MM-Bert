import argparse

import cmbert
from cmbert import SDK_DS
import model_config
import train

def main(args):
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

    preprocess = args.preprocess
    load_preprocessed = args.load_preprocessed


    dataset_config = cmbert.CmuDatasetConfig(
        sdkpath = r'D:\Studia\magisterskie\Praca_magisterska\data\repo\CMU-MultimodalSDK',
        dataset = ds,
        text_features = text_features,
        audio_features = audio_features,
        visual_features = visual_features,
        labels = labels,
        preprocess = preprocess,
        load_preprocessed = load_preprocessed,
    )

    training_arguments = train.TrainingArgs(
        batch_size = model_config.batch_size,
        num_epochs = model_config.num_epochs,
        criterion = model_config.criterion,
        optimizer = model_config.optimizer,
        lr = model_config.lr,
        scheduler_type = model_config.scheduler_type,
        num_warmup_steps = model_config.num_warmup_steps,
        best_model_metric = model_config.best_model_metric,
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
    )

    # for debug
    for attr, value in dataset_config.__dict__.items():
        print(f'{attr}: {value}')

    for attr, value in training_arguments.__dict__.items():
        print(f'{attr}: {value}')

    for attr, value in _model_config.__dict__.items():
        print(f'{attr}: {value}')
    # for debug

    train.main(
        dataset_config=dataset_config,
        model_config=_model_config,
        training_arguments=training_arguments,
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

    parser.add_argument('--ds', metavar='dataset', type=str,
                        choices=['cmumosi', 'cmumosei', 'pom'],
                        required=True,
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
                        action='store_false',
                        help='do preprocessing dataset, it takse long time - better reuse saved preprocessed data'
                        )
    
    parser.add_argument('--loadpreprocessed', dest='load_preprocessed',
                        action='store_true',
                        help='load preprocessed data - saved preprocessed data required'
                        )

    args = parser.parse_args()
    print(args)

    main(args)

