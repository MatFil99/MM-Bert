base_config = {
# model hyperparameters
'dataset': 'cmumosi',
'text_features': 'RAWTEXT',
# 'audio_features': 'COVAREP', # default 
# 'visual_features': 'FACET42', # default
'model_name': 'mmbert',
'encoder_checkpoint': 'distilbert/distilbert-base-uncased',
'hidden_dropout_prob': 0.2,
'modality_att_dropout_prob': 0.3,
'freeze_params_layers': 4,
'hidden_size': 768,
'projection_size': 30,

# training arguments
'batch_size': 8 ,
'num_epochs': 4,
'patience': 1,
'chunk_size': None,
'criterion': 'mseloss', # ['mseloss', 'maeloss']
'optimizer': 'adamw', # ['adamw', 'adam', 'rmsprop'] 'adadelta'
'layer_specific_optimization': True,
'lr': 2e-5 ,
'scheduler_type': 'linear', # ['linear', 'constant', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
'warmup_steps_ratio': 0.0 ,
'best_model_metric': 'mse', # ['mse', 'mae', 'pearsonr', 'loss']
'save_best_model': True,
'save_model_dest': 'models/reg'
}