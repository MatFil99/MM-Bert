multirun_configuration = {
# dataset setting 
'datasets': ['cmumosi'] , # 'cmumosei' 'cmumosi'
'text_features': ['RAWTEXT'],
'audio_features': ['COVAREP'],
'visual_features': ['FACET42'] ,

# model hyperparameters
'model_names': ['mmbert'],
'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
'hidden_dropout_prob': [0.2],
'modality_att_dropout_prob': [0.3],
# 'freeze_params': [True],
'freeze_params_layers': [4],
'hidden_size': [768] ,
'projection_size': [30],
'num_labels': [1] ,

# training arguments
'batch_size': [8],
'num_epochs': [4], # 
'patience': [1],
'chunk_size': [None],
'criterion': ['mseloss'],
'optimizer': ['adamw'] ,
'layer_specific_optimization': [False, True],
'lr': [2e-5] , # , 2e-5, 4e-5]
'scheduler_type': ['linear'] , # ['linear', 'constant'] # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
'warmup_steps_ratio': [0.0] ,
'best_model_metric': ['mae'], #, 'mae'] # ['mse', 'pearsonr', 'loss']

'save_best_model': [True],
'save_model_dest': ['models/reg'],
}
