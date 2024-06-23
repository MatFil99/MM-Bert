multirun_configuration = {
# dataset setting 
'datasets': ['cmumosei'] , # 'cmumosei' 'cmumosi'
'text_features': ['RAWTEXT'],
'audio_features': ['COVAREP'],
'visual_features': ['FACET42'],

# model hyperparameters
'model_names': ['mmbert'], #'cmbert'
'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
'hidden_dropout_prob': [0.2],
'modality_att_dropout_prob': [0.3],
# 'freeze_params': [True],
'freeze_params_layers': [4],
'hidden_size': [768] ,
'projection_size': [30],
'num_labels': [1] ,

# training arguments
'batch_size': [8] ,
'num_epochs': [20],
'patience': [2],
'chunk_size': [128],
'wwm_probability': [0.2],
'optimizer': ['adamw'] ,
'layer_specific_optimization': [True],
'lr': [2e-5] ,
'scheduler_type': ['linear'] ,
'warmup_steps_ratio': [0.0] ,
'best_model_metric': ['accuracy'],

'save_best_model': [True],
'save_model_dest': ['models/mlm'],
}