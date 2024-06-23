from transformers import DistilBertConfig


class MMBertConfig(DistilBertConfig):

    def __init__(self,
                 encoder_checkpoint = 'google-bert/bert-base-uncased',
                 hidden_dropout_prob = 0.1,
                 modality_att_dropout_prob = 0.1,
                #  freeze_params = True,
                 freeze_params_layers = 10,
                 hidden_size = 768,
                 audio_feat_size = 74,
                 visual_feat_size = 35,
                 projection_size = 30,
                 num_labels = 2,
                 best_model_metric = 'accuracy',
                 ):
        super().__init__()
        self.encoder_checkpoint = encoder_checkpoint
        self.hidden_dropout_prob = hidden_dropout_prob
        self.modality_att_dropout_prob = modality_att_dropout_prob
        # self.freeze_params = freeze_params
        self.freeze_params_layers = freeze_params_layers
        self.hidden_size = hidden_size
        self.audio_feat_size = audio_feat_size
        self.visual_feat_size = visual_feat_size
        self.projection_size = projection_size
        self.num_classes = num_labels
        self.best_model_metric = best_model_metric

