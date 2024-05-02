from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np

# not flexible model
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedModel, BertConfig, BertPreTrainedModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# checkpoint = 'google-bert/bert-base-uncased'


class CMBertConfig(BertConfig):

    def __init__(self,
                 encoder_checkpoint = 'google-bert/bert-base-uncased',
                 hidden_dropout_prob = 0.1,
                 modality_att_dropout_prob = 0.1,
                 hidden_size = 768,
                 audio_feat_size = 74,
                 visual_feat_size = 35,
                 projection_size = 30,
                 num_labels = 2,
                 ):
        super().__init__()
        self.encoder_checkpoint = encoder_checkpoint
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.audio_feat_size = audio_feat_size
        self.visual_feat_size = visual_feat_size
        self.projection_size = projection_size
        self.num_labels = num_labels

class ModalityAttention(nn.Module):

    def __init__(self, config):
        super(ModalityAttention, self).__init__()

        self.proj_t = nn.Conv1d(config.hidden_size, config.projection_size, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(config.audio_feat_size, config.projection_size, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(74, 30, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ReLU()
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, audio_data, attention_mask):
        att_mask = 1 - attention_mask.unsqueeze(1) # masked positions reprezented as 1, other 0 (vertical attention)
        att_mask_ = att_mask.permute(0, 2, 1) # horizontal attention mask
        att_mask_final = torch.logical_or(att_mask, att_mask_) * -10000.0 # square matrix attention

        text_data = hidden_states
        text_data = text_data.transpose(1,2) # prepare for conv1d
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1,2) # back to previous shape
        text_data_1 = text_data.reshape(-1).detach() # text_data.reshape(-1).cpu().detach().numpy() # ?
        weights = torch.sqrt(torch.norm(text_data_1, p=2))
        text_data = text_data / weights
        
        audio_data = audio_data.transpose(-1,-2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(-1,-2)
        # normalization was added
        audio_data_1 = audio_data.reshape(-1).detach() # 
        weights = torch.sqrt(torch.norm(audio_data_1, p=2))

        audio_data = audio_data / weights

        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att1 = self.activation(text_att)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        text_weight_1 = self.text_weight_1
        audio_weight_1 = self.audio_weight_1
        bias = self.bias

        fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att + bias
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

        return hidden_states_new, text_att1, fusion_att1


# class CMBertForSequenceClassification(BertPreTrainedModel):
class CMBertForSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config): #, num_labels=2):
        super(CMBertForSequenceClassification, self).__init__(config)

        print(config)

        self.num_labels = config.num_labels
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.encoder_checkpoint)
        # self.distilbert = AutoModel.from_pretrained(checkpoint)
        
        if config.audio_feat_size or config.visual_feat_size:
            self.modality_fusion = ModalityAttention(config)
        else:
            self.modality_fusion = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        audio_data=None,
        visual_data=None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ): # -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        if self.config.encoder_checkpoint == 'distilbert/distilbert-base-uncased':
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]
            pooled_output = outputs[0][:,0] # last of hidden state
        elif self.config.encoder_checkpoint == 'google-bert/bert-base-uncased':
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]
            pooled_output = outputs[0][:,0]
            # pooled_output = outputs[1]

        if self.modality_fusion:
            hidden_states, text_att, fusion_att = self.modality_fusion(hidden_states, audio_data, attention_mask)
    
        pooled_output = hidden_states[:,0] # 

        pooled_output = self.dropout(pooled_output)

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output) #

        # pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return {'logits': logits,'text_att': None,'fusion_att': None}
        # , text_att, fusion_att
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == '__main__':
    print('model.py')
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
    
    # print(sentences)
    print(tokenized['audio_data'].shape)
    print(tokenized['input_ids'].shape)


    output, _, _ = classifier(**tokenized)
    print(output)