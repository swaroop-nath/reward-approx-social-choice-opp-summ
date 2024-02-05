import json

######--------CHANGEABLE VARS--------######
training_mode = "supervised" # One of `supervised` or `limited-trajectory-rl`
sweep_name = training_mode

######--------SWEEP CONFIGURATION--------######
SWEEP_CONFIGURATION = {
    "name": sweep_name,
    "method": 'bayes',
    "metric": {
        "name": 'eval/loss',
        "goal": 'minimize'
    },
    "parameters": {
        "max-epochs": {
            "values": [7, 10, 13],
        },
        "grad-acc": {
            "values": [4, 8, 16],    
        },
        "learning-rate": {
            "distribution": 'uniform',
            "max": 5e-5,
            "min": 5e-6
        },
        "warmup-steps-frac": {
            "distribution": 'uniform',
            "min": 0.20,
            "max": 0.40
        }
    }
}

class Configuration:
    def __init__(self):
        ######--------DATA VARS--------######
        self.TRAIN_DATA_PATH = './instruction-data-v2/training-scored-final.jsonl'
        self.VALID_DATA_PATH = './instruction-data-v2/validation-scored.jsonl'
        self.TEST_DATA_PATH = './amazon_test_data.csv'
        self.SCORING_MODE = 'naive-mean' # `naive-mean`, `synthetic-feedback`, `dpo-baseline`, or `inductive-bias`
        self.OUTPUT_DIR = None
        self.TOTAL_INSTANCES = 20763
        
        ######--------MODEL VARS--------######
        self.BACKBONE_NAME = "facebook/bart-large"
        self.MODEL_PRETRAINED_PATH = None
        self.GEN_KWARGS = {'top_p': 0.90, 'top_k': 10, 'do_sample': True, 'max_new_tokens': 150}
        
        ######--------TRAINING VARS--------######
        self.TRAINING_MODE = training_mode
        self.SUPERVISED_LOSS_WEIGHTAGE = 0.25
        self.GRAD_ACC = None
        self.OPTIM_NAME = 'adamw_torch'
        self.LEARNING_RATE = None
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = None
        self.EPOCHS = None
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 64
        self.MAX_GRAD_NORM = 1.0
        self.USE_BF16 = False
        self.USE_FP16 = True
        self.WEIGHT_DECAY = 0.05
        
        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 10 # Keeping this same for every run is uniform juxtaposition
        self.EVAL_STEPS = 200
    
    def set_configuration_hparams(self, config):
        self.EPOCHS = config['max-epochs']
        self.GRAD_ACC = config['grad-acc']
        self.LEARNING_RATE = config['learning-rate']
        
        warmup_steps_frac = config['warmup-steps-frac']
        warmup_steps = int(self.TOTAL_INSTANCES * self.EPOCHS / (self.TRAIN_BATCH_SIZE * self.GRAD_ACC) * warmup_steps_frac)
        self.LR_WARMUP = warmup_steps
        
        return f"epoch-{self.EPOCHS}--grad-acc-{self.GRAD_ACC}--lr-{self.LEARNING_RATE:0.4f}--warmup-steps-frac-{warmup_steps_frac:0.4f}"
        
    def set_output_dir(self, output_dir):
        self.OUTPUT_DIR = output_dir
    
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
        }
        
        return json.dumps(serialization_string)