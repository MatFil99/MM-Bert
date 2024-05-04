# dataset setting
datasets = ['cmumosi'] # , 'pom', 'cmumosei']
text_features = ['RAWTEXT']
audio_features = ['COVAREP', None]
visual_features = ['FACET42', None] # ['FACET42', 'OPENFACE2']

# model hyperparameters
encoder_checkpoints = ['google-bert/bert-base-uncased'] # 'distilbert/distilbert-base-uncased' , 'google-bert/bert-base-uncased']
hidden_dropout_prob = [0.1, 0.2]
modality_att_dropout_prob = [0.3]
hidden_size = [768] # depends on chosen encoder, for these it is the same
projection_size = [30, 768] # , 90] #
num_labels = [1, 2, 7] 

# training arguments
batch_size = [8] # [8, 16, 32]
num_epochs = [3] # [3, 4]
criterion = {
    'classification': ['crossentropyloss'], 
    'regression': ['mseloss'] #, 'maeloss']
    }
optimizer = ['adamw'] # ['adamw', 'adam', 'rmsprop']
lr = [2e-5] # , 2e-5, 4e-5]
scheduler_type = ['linear'] # ['linear', 'constant'] # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
warmup_steps_ratio = [0.0] #
best_model_metric = {
    'classification': ['accuracy'], # ['accuracy', 'f1', 'loss']
    'regression': ['mse'] #, 'mae'] # ['mse', 'pearsonr', 'loss']
}
save_best_model = [False]
save_model_dest = ['models/']