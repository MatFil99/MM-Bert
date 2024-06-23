# define parameters to test ()
multirun_configuration = {
    
    'hidden_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
    'modality_att_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
    'projection_size': [30, 60, 120, 240, 480, 768],
    'freeze_params_layers': [0, 3, 4, 5, 6],

# other possible parameters
# # dataset setting # remove it

# 'text_features': ['RAWTEXT'],
# 'audio_features': ['COVAREP'],
# 'visual_features': ['FACET42'],

# # model hyperparameters
# 'model_names': ['mmbert'],
# 'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
# 'hidden_dropout_prob': [0.2],
# 'modality_att_dropout_prob': [0.3],
# # 'freeze_params': [True],
# 'freeze_params_layers': [4],
# 'hidden_size': [768] ,
# 'projection_size': [30],
# 'num_labels': [2] ,

# # training arguments
# 'batch_size': [8] ,
# 'num_epochs': [4] ,
# 'patience': [1],
# 'chunk_size': [None],
# 'criterion': ['crossentropyloss'],
# 'optimizer': ['adamw'] ,
# 'layer_specific_optimization': [True],
# 'lr': [2e-5] ,
# 'scheduler_type': ['linear'] ,
# 'warmup_steps_ratio': [0.0] ,
# 'best_model_metric': ['accuracy'],

# 'save_best_model': [True],
# 'save_model_dest': ['models/class'],
}


