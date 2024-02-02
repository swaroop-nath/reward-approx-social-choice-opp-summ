import json

######--------CHANGEABLE VARS--------######
training_mode = "limited-trajectory-rl" # One of `supervised` or `limited-trajectory-rl`

class Configuration:
    def __init__(self):
        ######--------DATA VARS--------######
        self.TRAIN_DATA_PATH = './instruction-data-v2/training-scored.jsonl'
        self.VALID_DATA_PATH = './instruction-data-v2/validation-scored.jsonl'
        self.TEST_DATA_PATH = './amazon_test_data.csv'
        self.SCORING_MODE = 'naive-mean' # `naive-mean`, `synthetic-feedback`, or `human-feedback`, `` (empty) for supervised
        if self.SCORING_MODE == '': self.OUTPUT_DIR = f'./run-{training_mode}'
        else: self.OUTPUT_DIR = f'./run-{training_mode}-{self.SCORING_MODE}'
        
        ######--------MODEL VARS--------######
        self.BACKBONE_NAME = "facebook/bart-large"
        self.MODEL_PRETRAINED_PATH = None
        self.GEN_KWARGS = {'top_p': 0.90, 'top_k': 10, 'do_sample': True, 'max_new_tokens': 150}
        
        ######--------TRAINING VARS--------######
        self.TRAINING_MODE = training_mode
        self.SUPERVISED_LOSS_WEIGHTAGE = 0.05
        self.GRAD_ACC = 4
        self.OPTIM_NAME = 'adamw_torch'
        self.LEARNING_RATE = 1e-5
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = 200
        self.EPOCHS = 20
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 16
        self.MAX_GRAD_NORM = 1.0
        self.USE_BF16 = False
        self.USE_FP16 = True
        self.WEIGHT_DECAY = 0.05
        
        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 2 * self.GRAD_ACC
        self.EVAL_STEPS = 200
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