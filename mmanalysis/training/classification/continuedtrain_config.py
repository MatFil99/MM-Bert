contrain_configuration = {
# dataset setting 
'datasets': ['cmumosei'] ,
'text_features': ['RAWTEXT'],
'audio_features': ['COVAREP'], # values depends on pretrained model
'visual_features': ['FACET42'] , # values depends on pretrained model

# model hyperparameters
'model_names': [None], # values depends on pretrained model
'encoder_checkpoints': ['distilbert/distilbert-base-uncased'] ,
'hidden_dropout_prob': [0.1],
'modality_att_dropout_prob': [0.3],
'freeze_params_layers': [3],
'hidden_size': [768] , # values depends on pretrained model
'projection_size': [None] , # values depends on pretrained model
'num_labels': [2] ,

# training arguments
'batch_size': [8] ,
'num_epochs': [4] ,
'patience': [3],
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
