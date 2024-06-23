
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
'criterion': 'crossentropyloss',
'optimizer': 'adamw',
'layer_specific_optimization': True,
'lr': 2e-5,
'scheduler_type': 'linear',
'warmup_steps_ratio': 0.0,
'best_model_metric': 'accuracy',

'save_best_model': True,
# 'save_model_dest': 'models/class' # it is overwritten in code
}