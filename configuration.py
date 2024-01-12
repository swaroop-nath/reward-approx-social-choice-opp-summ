import json

######--------CHANGEABLE VARS--------######
training_mode = "supervised" # One of `supervised` or `limited-trajectory-rl`

class Configuration:
    def __init__(self):
        ######--------DATA VARS--------######
        self.TRAIN_DATA_PATH = None
        self.VALID_DATA_PATH = None
        self.OUTPUT_DIR = f'./run-{training_mode}'
        
        ######--------MODEL VARS--------######
        self.BACKBONE_NAME = "facebook/bart-large"
        
        ######--------TRAINING VARS--------######
        self.TRAINING_MODE = training_mode
        self.SUPERVISED_LOSS_WEIGHTAGE = 0.05
        self.GRAD_ACC = 2
        self.OPTIM_NAME = 'adam'
        self.LEARNING_RATE = 1e-5
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = 2000
        self.EPOCHS = 10
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 8
        self.MAX_GRAD_NORM = 1.0
        self.USE_BF16 = True
        self.USE_FP16 = False
        
        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 1000
        self.EVAL_STEPS = 2000
        self.RUN_NAME = None
    
    def set_run_name(self, run_name):
        self.RUN_NAME = run_name 
        
    def serialize(self):
        serialization_string = {
            'back-bone': self.BACKBONE_NAME,
            'training-mode': self.TRAINING_MODE,
            'supervised-loss-weights': self.SUPERVISED_LOSS_WEIGHTAGE,
            'grad-acc': self.GRAD_ACC,
            'optim-name': self.OPTIM_NAME,
            'learning-rate': self.LEARNING_RATE,
            'lr-scheduler': self.LR_SCHEDULER,
            'lr-warmup': self.LR_WARMUP,
            'epochs': self.EPOCHS,
            'use-fp16': self.USE_FP16,
            'use-bf16': self.USE_BF16,
            'grad-norm': self.MAX_GRAD_NORM,
            'train-batch-size': self.TRAIN_BATCH_SIZE,
            'eval-batch-size': self.EVAL_BATCH_SIZE,
            'log-steps': self.LOG_STEPS,
            'eval-steps': self.EVAL_STEPS,
            'run-name': self.RUN_NAME
        }
        
        return json.dumps(serialization_string)