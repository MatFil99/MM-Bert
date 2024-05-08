singlerun_configuration = {
# model hyperparameters
'model_name': 'mmbert',
'encoder_checkpoint': 'distilbert/distilbert-base-uncased' ,
'hidden_dropout_prob': 0.2 ,
'modality_att_dropout_prob': 0.3 ,
'hidden_size': 768 ,
'projection_size': 768 ,
'num_labels': None ,

# training arguments
'batch_size': 24 ,
'num_epochs': 3,
'chunk_size': 128,
'optimizer': 'adamw',
'lr': 5e-5 , # 2e-5
'scheduler_type': 'linear', # ['linear', 'constant', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
'warmup_steps_ratio': 0.0 ,
'save_best_model': True,
'save_model_dest': 'models/'
}
