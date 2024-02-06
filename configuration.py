import json
import torch

######--------CHANGEABLE VARS--------######
training_mode = "limited-trajectory-rl" # One of `supervised` or `limited-trajectory-rl`
sweep_name = training_mode
track_metric = 'rouge-L'
goal = 'max'
rl_algorithm = 'proximal-policy-optimization'

######--------SWEEP CONFIGURATION--------######
SWEEP_CONFIGURATION = {
    "name": sweep_name,
    "method": 'bayes',
    "metric": {
        "name": f'eval/best-{track_metric}',
        "goal": f"{goal}imize"
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
        self.SUPERVISED_LOSS_WEIGHTAGE = 0.0
        self.RL_ALGORITHM = rl_algorithm
        self.VALUE_HEAD_DROPOUT = 0.1
        self.OLD_MODEL_UPDATE_INTERVAL = 5
        self.KL_PENALTY_MODE = 'instruct-gpt'
        self.KL_BETA = 0.2
        self.GAMMA = 1
        self.GAE_LAMBDA = 0.95
        self.CLIP_LIM = 0.2
        self.GRAD_ACC = None
        self.OPTIM_NAME = 'adamw_torch'
        self.LEARNING_RATE = None
        self.LR_SCHEDULER = 'cosine'
        self.LR_WARMUP = None
        self.EPOCHS = None
        self.TRAIN_BATCH_SIZE = 8
        self.EVAL_BATCH_SIZE = 32
        self.MAX_GRAD_NORM = 1.0
        self.USE_BF16 = False
        self.USE_FP16 = True
        self.WEIGHT_DECAY = 0.05
        
        ######--------LOGGING VARS--------######
        self.LOG_STEPS = 10 # Keeping this same for every run is uniform juxtaposition
        self.EVAL_STEPS = 200
        self.TRACK_METRIC = track_metric
        self.GOAL = goal
    
    def set_configuration_hparams(self, config):
        self.EPOCHS = config['max-epochs']
        self.GRAD_ACC = config['grad-acc']
        self.LEARNING_RATE = config['learning-rate']
        
        warmup_steps_frac = config['warmup-steps-frac']
        warmup_steps = int(self.TOTAL_INSTANCES * self.EPOCHS / (self.TRAIN_BATCH_SIZE * self.GRAD_ACC * torch.cuda.device_count()) * warmup_steps_frac)
        print(f"Setting warmup to ({self.TOTAL_INSTANCES} x {self.EPOCHS} * {warmup_steps_frac}) / ({self.TRAIN_BATCH_SIZE} * {self.GRAD_ACC} * {torch.cuda.device_count()}) = {warmup_steps}")
        self.LR_WARMUP = warmup_steps
        
        return f"epoch-{self.EPOCHS}--grad-acc-{self.GRAD_ACC}--lr-{self.LEARNING_RATE:0.4g}--warmup-steps-frac-{warmup_steps_frac:0.4g}"
        
    def set_output_dir(self, output_dir):
        self.OUTPUT_DIR = output_dir
    
    def serialize(self):
        return json.dumps(self.__dict__)