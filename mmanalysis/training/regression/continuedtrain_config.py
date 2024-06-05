contrain_configuration = {
# dataset setting 
'datasets': ['pom'] , # 'cmumosei'
'text_features': ['RAWTEXT'],
'audio_features': [None], # values depends on pretrained model
'visual_features': [None] , # values depends on pretrained model

# model hyperparameters
'model_names': [None], # values depends on pretrained model
'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
'hidden_dropout_prob': [0.2],
'modality_att_dropout_prob': [0.3],
'freeze_params': [True],
'hidden_size': [768] , # values depends on pretrained model
'projection_size': [None] , # values depends on pretrained model
'num_labels': [1] ,

# training arguments
'batch_size': [8],
'num_epochs': [4], # 
'patience': [3],
'chunk_size': [None],
'criterion': ['mseloss'],
'optimizer': ['adamw'] ,
'layer_specific_optimization': [True],
'lr': [2e-5] , # , 2e-5, 4e-5]
'scheduler_type': ['linear'] , # ['linear', 'constant'] # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
'warmup_steps_ratio': [0.0] ,
'best_model_metric': ['mae'], #, 'mae'] # ['mse', 'pearsonr', 'loss']

'save_best_model': [True],
'save_model_dest': ['models/reg'],
}
