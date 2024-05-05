from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

# not flexible model
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedModel, BertConfig, BertPreTrainedModel, DistilBertPreTrainedModel, DistilBertConfig
from transformers.activations import GELUActivation
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput


# checkpoint = 'google-bert/bert-base-uncased'


class CMBertConfig(DistilBertConfig):
# class CMBertConfig(BertConfig):

    def __init__(self,
                 encoder_checkpoint = 'google-bert/bert-base-uncased',
                 hidden_dropout_prob = 0.1,
                 modality_att_dropout_prob = 0.1,
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
        self.hidden_size = hidden_size
        self.audio_feat_size = audio_feat_size
        self.visual_feat_size = visual_feat_size
        self.projection_size = projection_size
        self.num_classes = num_labels
        self.best_model_metric = best_model_metric

    

class ModalityAttention(nn.Module):

    def __init__(self, config):
        super(ModalityAttention, self).__init__()

        self.proj_t = nn.Conv1d(config.hidden_size, config.projection_size, kernel_size=1, padding=0, bias=False)
        
        if config.audio_feat_size is not None:
            self.audio_modality = True
            self.proj_a = nn.Conv1d(config.audio_feat_size, config.projection_size, kernel_size=1, padding=0, bias=False)
            self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.audio_weight_1.data.fill_(1)
        else:
            self.audio_modality = False
            self.proj_a = None
            self.audio_weight_1 = None

        if config.visual_feat_size is not None:
            self.visual_modality = True
            self.proj_v = nn.Conv1d(config.visual_feat_size, config.projection_size, kernel_size=1, padding=0, bias=False)
            self.visual_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.visual_weight_1.data.fill_(1)
        else:
            self.visual_modality = False
            self.proj_v = None
            self.visual_weight_1 = None

        self.activation = nn.ReLU()
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.text_weight_1.data.fill_(1)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout1 = nn.Dropout(config.modality_att_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, audio_data, visual_data, attention_mask):
        att_mask = 1 - attention_mask.unsqueeze(1) # masked positions reprezented as 1, other 0 (vertical attention)
        att_mask_ = att_mask.permute(0, 2, 1) # horizontal attention mask
        att_mask_final = torch.logical_or(att_mask, att_mask_) * -10000.0 # square matrix attention

        # text projection
        text_data = hidden_states
        text_data = text_data.transpose(1,2) # prepare for conv1d
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1,2) # back to previous shape
        text_data_1 = text_data.reshape(-1).detach() # text_data.reshape(-1).cpu().detach().numpy() # ?
        weights = torch.sqrt(torch.norm(text_data_1, p=2))
        text_data = text_data/weights
        
        if self.audio_modality:
            # audio projection
            audio_data = audio_data.transpose(1,2)
            audio_data = self.proj_a(audio_data)
            audio_data = audio_data.transpose(1,2)

            # normalization was added
            audio_data_1 = audio_data.reshape(-1).detach() # 
            weights = torch.sqrt(torch.norm(audio_data_1, p=2))
            audio_data = audio_data / weights

            # attention
            text_x_audio_att = torch.matmul(text_data, audio_data.transpose(-1, -2))
            text_x_audio_att = self.activation(text_x_audio_att)
            text_x_audio_att_weighted = text_x_audio_att * self.audio_weight_1
        else:
            text_x_audio_att_weighted = 0

        if self.visual_modality:
            # visual projection
            visual_data = visual_data.transpose(-1,-2)
            visual_data = self.proj_v(visual_data)
            visual_data = visual_data.transpose(-1,-2)
            # normalization was added
            visual_data_1 = visual_data.reshape(-1).detach() # 
            weights = torch.sqrt(torch.norm(visual_data_1, p=2))
            visual_data = visual_data / weights

            # attention
            text_x_visual_att = torch.matmul(text_data, visual_data.transpose(-1, -2))
            text_x_visual_att = self.activation(text_x_visual_att)
            text_x_visual_att_weighted = text_x_visual_att * self.visual_weight_1
        else:
            text_x_visual_att_weighted = 0


        # text attention
        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att = self.activation(text_att)
        text_att_weighted = text_att * self.text_weight_1

        bias = self.bias

        # fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att + visual_weight_1 * visual_att + bias
        fusion_att = text_x_audio_att_weighted + text_x_visual_att_weighted + bias
        # fusion_att = text_att_weighted + text_x_audio_att_weighted + text_x_visual_att_weighted + bias
        
        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + att_mask_final
        fusion_att = self.softmax(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        # there could be used other approaches
        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        # norm
        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.norm(hidden_states_new)

        hidden_states_new = fusion_data

        return hidden_states_new, text_att, fusion_att1

class CMBertForMaskedLM(DistilBertPreTrainedModel):

    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.activation = GELUActivation() # GELUActivation # nn.GELU

        self.encoder = AutoModel.from_pretrained(config.encoder_checkpoint)
        
        if config.audio_feat_size or config.visual_feat_size:
            self.modality_fusion = ModalityAttention(config)
        else:
            self.modality_fusion = None
        
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()
    
    def forward(
        self,
        audio_data=None,
        visual_data=None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs[0]
        text_att = None
        fusion_att = None
        if self.modality_fusion:
            hidden_states, text_att, fusion_att = self.modality_fusion(hidden_states, audio_data, visual_data, attention_mask)

        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=outputs[0],
            # attentions=,
        )
    # mlm_loss, prediction_logits, outputs[0]

# class CMBertForSequenceClassification(PreTrainedModel):
class CMBertForSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config): #, num_labels=2):
        super(CMBertForSequenceClassification, self).__init__(config)

        self.config = config

        self.encoder = AutoModel.from_pretrained(config.encoder_checkpoint)
        if config.audio_feat_size or config.visual_feat_size:
            self.modality_fusion = ModalityAttention(config)
        else:
            self.modality_fusion = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        # Initialize weights and apply final processing
        
        self.post_init()
        if self.modality_fusion:
            self.freeze_params()

    def forward(
        self,
        audio_data=None,
        visual_data=None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ): # -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]

        text_att = None
        fusion_att = None
        if self.modality_fusion:
            hidden_states, text_att, fusion_att = self.modality_fusion(hidden_states, audio_data, visual_data, attention_mask)
    
        pooled_output = hidden_states[:,0] # 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return {'logits': logits,'text_att': text_att,'fusion_att': fusion_att}

    def freeze_params(self):
        if self.config.encoder_checkpoint == 'google-bert/bert-base-uncased':
            encoder_layer = 'encoder.layer.'
        else:
            encoder_layer = 'transformer.layer.'

        for name, param in self.named_parameters():
            param.requires_grad = False
            for i in range(12):
                if encoder_layer + str(i) in name:
                    param.requires_grad = True
            if 'modality_fusion' in name or 'pooler' in name:
                param.requires_grad = True

if __name__ == '__main__':
    cmbertconfig = CMBertConfig(
        hidden_dropout_prob = 0.1
        )
    num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(cmbertconfig.encoder_checkpoint)
    classifier = CMBertForSequenceClassification(cmbertconfig, num_labels)


    sentences = [
        'this is the world we live in',
        'each Marvel\'s movie is great, but this one leaves a lot to be desired'
    ]

    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    tokenized['audio_data'] = torch.ones(tokenized['input_ids'].shape[0], tokenized['input_ids'].shape[1], 74)
    

    output, _, _ = classifier(**tokenized)