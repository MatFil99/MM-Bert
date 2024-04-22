import torch
from torch import nn
import numpy as np

# not flexible model
from transformers import AutoModel, AutoTokenizer
checkpoint = 'google-bert/bert-base-uncased'


class CMBertConfig():

    def __init__(self,
                 encoder_checkpoint = 'google-bert/bert-base-uncased',
                 hidden_dropout_prob = 0.1,
                 hidden_size = 768
                 ):
        self.encoder_checkpoint = encoder_checkpoint
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size

class ModalityAttention(nn.Module):

    def __init__(self, config):
        super(ModalityAttention, self).__init__()
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(74, 30, kernel_size=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(768)

    def forward(self, hidden_states, audio_data, attention_mask):
        att_mask = 1 - attention_mask.unsqueeze(1) # masked positions reprezented as 1, other 0 (vertical attention)
        att_mask_ = att_mask.permute(0, 2, 1) # horizontal attention
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


class CMBertForSequenceClassification(nn.Module):

    def __init__(self, config, num_labels):
        super(CMBertForSequenceClassification, self).__init__()

        self.encoder = AutoModel.from_pretrained(config.encoder_checkpoint)
        self.modality_fusion = ModalityAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, audio_data, token_type_ids=None, attention_mask=None, labels=None):
        (_, hidden_states), (_, pooled_output) = self.encoder(input_ids, token_type_ids, attention_mask).items()
        
        hidden_states, text_att, fusion_att = self.modality_fusion(hidden_states, audio_data, attention_mask)
        pooled_output = hidden_states[:,0] # 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, text_att, fusion_att

        

if __name__ == '__main__':
    print('model.py')
    cmbertconfig = CMBertConfig(
        hidden_dropout_prob = 0.2
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
    # print(tokenized['audio_data'].shape)

    output, _, _ = classifier(**tokenized)
    print(output)