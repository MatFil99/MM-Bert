singlerun_configuration = {
# model hyperparameters
'model_name': 'cmbert',
'encoder_checkpoint': 'distilbert/distilbert-base-uncased' ,
'hidden_dropout_prob': 0.2 ,
'modality_att_dropout_prob': 0.3 ,
'freeze_params': True,
'hidden_size': 768 ,
'projection_size': 768,
'num_labels': None,

# training arguments
'batch_size': 8 ,
'num_epochs': 40,
'patience': 2,
'chunk_size': 128,
'wwm_probability': 0.2,
'criterion': 'crossentropyloss', # remove
'optimizer': 'adamw',
'layer_specific_optimization': False,
'lr': 5e-5 , # 2e-5
'scheduler_type': 'linear', # ['linear', 'constant', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
'warmup_steps_ratio': 0.0 ,
'best_model_metric': 'accuracy', # remove
'save_best_model': True,
'save_model_dest': 'models/mlm'
}
