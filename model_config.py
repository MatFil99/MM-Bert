# model hyperparameters
encoder_checkpoint = 'google-bert/bert-base-uncased'
hidden_dropout_prob = 0.2
modality_att_dropout_prob = 0.1
hidden_size = 768 # depends on chosen encoder
projection_size = 30
num_labels = 2

# training arguments
batch_size = 8
num_epochs = 3
criterion = 'crossentropyloss'
optimizer = 'adamw'
lr = 2e-5
scheduler_type = 'linear'
num_warmup_steps = 0
best_model_metric = 'accuracy'