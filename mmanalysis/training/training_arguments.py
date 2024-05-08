

# default values for classification
class TrainingArgs():

    def __init__(self,
                 batch_size = 8,
                 num_epochs = 3,
                 criterion = 'crossentropy',
                 optimizer = 'adamw',
                 lr = 2e-5,
                 scheduler_type = 'linear',
                 warmup_steps_ratio = 0.0,
                 best_model_metric = 'accuracy',
                 higher_metric_best = True,
                 layer_specific_optimization = True,
                #  metrics = ['accuracy', 'f1'],
                 chunk_size = None, # for masked language modeling
                 best_model_path = None,
                 save_best_model = False,
                 save_model_dest = None,
                 ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.chunk_size = chunk_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.layer_specific_optimization = layer_specific_optimization
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps_ratio = warmup_steps_ratio
        self.best_model_metric = best_model_metric
        self.higher_metric_best = higher_metric_best
        self.save_best_model = save_best_model
        self.best_model_path = best_model_path
        self.save_model_dest = save_model_dest


