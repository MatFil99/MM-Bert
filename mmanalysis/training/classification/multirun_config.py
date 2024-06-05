multirun_configuration = {
# dataset setting 
'datasets': ['pom'] , # 'cmumosei' 'cmumosi'
'text_features': ['RAWTEXT'],
'audio_features': ['COVAREP', None],
'visual_features': ['FACET42', None],

# model hyperparameters
'model_names': ['cmbert', 'mmbert'],
'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
'hidden_dropout_prob': [0.1, 0.2],
'modality_att_dropout_prob': [0.3],
'freeze_params': [True],
'hidden_size': [768] ,
'projection_size': [30, 768],
'num_labels': [2, 7] ,

# training arguments
'batch_size': [8] ,
'num_epochs': [4] ,
'patience': [1],
'chunk_size': [None],
'criterion': ['crossentropyloss'],
'optimizer': ['adamw'] ,
'layer_specific_optimization': [True],
'lr': [2e-5] ,
'scheduler_type': ['linear'] ,
'warmup_steps_ratio': [0.0] ,
'best_model_metric': ['accuracy'],

'save_best_model': [True],
'save_model_dest': ['models/class'],
}
