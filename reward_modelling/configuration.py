import torch
import json

######--------CHANGEABLE VARS--------######
rm_train = 'synthetic-feedback'
use_bf16 = True
use_fp16 = (not use_bf16) and False
sweep_name = rm_train

######--------SWEEP CONFIG--------######
SWEEP_CONFIGURATION = {
    "name": sweep_name,
    "method": 'bayes',
    "metric": {
        "name": f'eval/loss',
        "goal": f"minimize"
    },
    "parameters": {
        "max-epochs": {
            "values": [100, 150, 200],
        },
        "activation": {
            "values": ["tanh", "relu", "sigmoid", None]
        },
        "layer-config": {
            "values": [
                {
                    "num-layers": 1,
                    "layer-wise-size": [5]
                },
                {
                    "num-layers": 1,
                    "layer-wise-size": [3]
                },
                {
                    "num-layers": 2,
                    "layer-wise-size": [5, 3]
                },
                {
                    "num-layers": 2,
                    "layer-wise-size": [4, 2]
                },
                {
                    "num-layers": 3,
                    "layer-wise-size": [6, 4, 2]
                }
            ]
        },
        "grad-acc": {
            "values": [1, 2, 4],    
        },
        "learning-rate": {
            "distribution": 'uniform',
            "max": 5e-2,
            "min": 5e-3
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
        self.TRAIN_DATA_PATH = '../instruction-data-v2/training-scored-final-cleaned.jsonl'
        self.VALID_DATA_PATH = '../instruction-data-v2/validation-scored-cleaned.jsonl'
        self.OUTPUT_DIR = None
        self.TRAIN_SAMPLE_SIZE = 20763
        self.VALID_SAMPLE_SIZE = 5000
        
        ######--------MODEL VARS--------######
        self.RM_TRAIN = rm_train
        self.LAYER_CONFIG = None
        self.NUM_INPUTS = 7
        self.NUM_OUTPUTS = 1
        self.ACTIVATION = None
        
        ######--------TRAINING VARS--------######
        self.TRAIN_BATCH_SIZE = 1024
        self.EVAL_BATCH_SIZE = 16
        self.GRAD_ACC = None
        self.EPOCHS = None
        self.LEARNING_RATE = None
        self.USE_BF16 = use_bf16
        self.USE_FP16 = use_fp16
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = None
        self.OPTIM_NAME = 'adamw_torch'
        self.WEIGHT_DECAY = 0.05
        self.MAX_GRAD_NORM = 1.0

        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 20 # Keeping this same for every run is uniform juxtaposition
        self.EVAL_STEPS = 200
        
    def set_configuration_hparams(self, config):
        self.EPOCHS = config['max-epochs']
        self.GRAD_ACC = config['grad-acc']
        self.LEARNING_RATE = config['learning-rate']
        
        warmup_steps_frac = config['warmup-steps-frac']
        warmup_steps = int(self.TRAIN_SAMPLE_SIZE * self.EPOCHS / (self.TRAIN_BATCH_SIZE * self.GRAD_ACC * torch.cuda.device_count()) * warmup_steps_frac)
        print(f"Setting warmup to ({self.TRAIN_SAMPLE_SIZE} x {self.EPOCHS} * {warmup_steps_frac}) / ({self.TRAIN_BATCH_SIZE} * {self.GRAD_ACC} * {torch.cuda.device_count()}) = {warmup_steps}")
        self.LR_WARMUP = warmup_steps
        self.LAYER_CONFIG = config['layer-config']
        self.ACTIVATION = config['activation']
        
        return f"epoch-{self.EPOCHS}--grad-acc-{self.GRAD_ACC}--lr-{self.LEARNING_RATE:0.4g}--warmup-steps-frac-{warmup_steps_frac:0.4g}"
    
    def set_output_dir(self, output_dir):
        self.OUTPUT_DIR = output_dir
    
    def serialize(self):
        return json.dumps(self.__dict__)