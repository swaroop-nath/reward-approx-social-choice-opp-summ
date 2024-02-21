from model_code import LimitedTrajectoryOpinionSummarizer
from configuration import Configuration, SWEEP_CONFIGURATION
from transformers import Trainer, TrainingArguments, TrainerCallback
from coolname import generate_slug
import wandb
import os
from tqdm import tqdm
import json
from utils import get_rouge_score
import torch
from data_handler import data_collator_fn_supervised, data_collator_fn_limited_trajectory, ReviewsDataset, ReviewsTestDataset
from shutil import rmtree
import numpy as np

##=========Custom Callback for Logging Setup=========##
class CustomTrainerCallback(TrainerCallback):
    def add_model(self, model, track_metric):
        self._model = model
        self._track_metric_name = track_metric
        self._track_metric_val = None
        self._epoch_counter = 0
        
    def update_track_metric(self, metric_val):
        self._track_metric_val = metric_val
        
    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self._epoch_counter += 1
        if self._epoch_counter > 2:
            self._model.update_ce_rl_trade_off()
        
    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        logs = self._model.get_logs()
        wandb_logs = {}
        for k, v in logs.items():
            if k not in ['loss']: wandb_logs['train/' + k] = v
        wandb.log(wandb_logs)
        self._model.update_parameters_on_step_end()
        
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        logs = self._model.get_logs()
        wandb_logs = {}
        for k, v in logs.items():
            wandb_logs['eval/' + k] = v
        wandb.log(wandb_logs)
        self._model.update_parameters_on_step_end()
        self._model.update_parameters_on_evaluate_end()
                
    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        wandb.log({f"eval/best-{self._track_metric_name}": self._track_metric_val})
        
##=========Custom Trainer for Logging Setup=========##
class CustomTrainer(Trainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None), preprocess_logits_for_metrics=None, track_metric='loss', goal='max', amazon_test_dataset=None, flipkart_test_dataset=None, oposum_test_dataset=None, output_dir_logging=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self._eval_logs = None
        self._best_model_sd = None
        self._goal = goal
        if self._goal == 'min': self._best_metric_val = float('inf')
        elif self._goal == 'max': self._best_metric_val = -float('inf')
        self._track_metric = track_metric
        self.amazon_test_dataset = amazon_test_dataset
        self.flipkart_test_dataset = flipkart_test_dataset
        self.oposum_test_dataset = oposum_test_dataset
        self.output_dir_logging = output_dir_logging
        
    def add_callback(self, callback):
        super().add_callback(callback)
        if isinstance(callback, CustomTrainerCallback): 
            callback.add_model(self.model, self._track_metric)
            self._custom_callback = callback
        
    def add_output_dir(self, dir):
        self.output_dir = dir
        
    def add_generation_kwargs(self, generation_kwargs):
        self.generation_kwargs = generation_kwargs
    
    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.run_on_test_dataset(self.model.get_max_length(), output_dir=f"{self.output_dir_logging}/step-{self.state.global_step}")
        self.model.eval()
        output_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix) # Does the normal evaluate, then followed by logging new metrics
        assert self._eval_logs is not None, f"eval-logs are None in evaluation"
        wandb_logs = {}
        for key, val in self._eval_logs.items():
            wandb_logs["eval/" + key] = np.mean(val)
        
        output_metrics.update(wandb_logs)
        wandb.log(wandb_logs)
        
        print(wandb_logs)
        if (self._goal == 'min' and wandb_logs["eval/" + self._track_metric] < self._best_metric_val) or \
            (self._goal == 'max' and wandb_logs["eval/" + self._track_metric] > self._best_metric_val):
            self._best_metric_val = wandb_logs["eval/" + self._track_metric]
            self._best_model_sd = self.model.state_dict()
            self._custom_callback.update_track_metric(self._best_metric_val)
        
        self._eval_logs = None # Setting it to None for new evaluation
            
        self.model.train()
        return output_metrics
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if self._eval_logs is None: self._eval_logs = {}
        tokenizer = self.model.get_tokenizer()
        
        if 'reviews-input-ids' in inputs: # Supervised
            batch = {'input_ids': inputs['reviews-input-ids'], 'attention_mask': inputs['reviews-attention-mask']}
            gt_summaries_ids = inputs['gt-summaries']
        elif 'sample-good' in inputs: # RL
            inputs = inputs['sample-good']
            batch = {'input_ids': inputs['reviews-input-ids'], 'attention_mask': inputs['reviews-attention-mask']}
            gt_summaries_ids = inputs['gt-summaries']
        else: raise NotImplementedError("Error in Prediction Loop -- can't find good input type")
        
        batch = self._prepare_inputs(batch)
        with torch.no_grad():
            outputs = self.model.generate(batch, **{'do_sample': False, 'num_beams': 1, 'max_new_tokens': 100})
        gen_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        gt_summaries = tokenizer.batch_decode(gt_summaries_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        scores = get_rouge_score(gen_summaries, gt_summaries)
        for score_key, score_val in scores.items():
            if score_key not in self._eval_logs: self._eval_logs[score_key] = []
            self._eval_logs[score_key].append(score_val)
            
        return loss, logits, labels
    
    def _run_through_test_set(self, model, tokenizer, test_dataset, max_length, fname, output_dir):
        pbar = tqdm(total=len(test_dataset), desc='Running on test set')
        writeable = []
        true_summaries, pred_summaries = [], []
        for unique_id, review_text, summary in test_dataset:
            review_input_ids = tokenizer([review_text], return_tensors='pt')['input_ids']
            batch = {'input_ids': review_input_ids[:, :max_length]}
            batch = self._prepare_inputs(batch)
            output = model.generate(batch, **self.generation_kwargs)[0]
            gen_summary = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            true_summaries.append(summary)
            pred_summaries.append(gen_summary)
            
            writeable.append(json.dumps({'unique-id': str(unique_id), 'review-text': review_text, 'gt-summary': summary, 'pred-summary': gen_summary}))
            pbar.update(1)
            
        pbar.close()
        
        with open(f"{output_dir}/{fname}.jsonl", 'w') as file:
            file.write('\n'.join(writeable))
        
        return true_summaries, pred_summaries
        
    def run_on_test_dataset(self, max_length, output_dir):
        tokenizer = self.model.get_tokenizer()
        # best_model_sd = self._best_model_sd
        # self.model.load_state_dict(best_model_sd)
        print('Loaded best model state dict, predicting on the test set')
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        true_summaries, pred_summaries = self._run_through_test_set(self.model, tokenizer, self.amazon_test_dataset, max_length, 'amazon', output_dir)        
        amazon_scores = get_rouge_score(pred_summaries, true_summaries)
        with open(f"{output_dir}/amazon-scores.json", 'w') as file:
            json.dump(amazon_scores, file)
            
        true_summaries, pred_summaries = self._run_through_test_set(self.model, tokenizer, self.flipkart_test_dataset, max_length, 'flipkart', output_dir)        
        flipkart_scores = get_rouge_score(pred_summaries, true_summaries)
            
        with open(f"{output_dir}/flipkart-scores.json", 'w') as file:
            json.dump(flipkart_scores, file)
            
        true_summaries, pred_summaries = self._run_through_test_set(self.model, tokenizer, self.oposum_test_dataset, max_length, 'oposum', output_dir)        
        oposum_scores = get_rouge_score(pred_summaries, true_summaries)
            
        with open(f"{output_dir}/oposum-scores.json", 'w') as file:
            json.dump(oposum_scores, file)
            
        print('Saving best model . . . ')
        self.save_model()
        
##=========WandB Setup=========##
wandb.login()
os.environ['WANDB_PROJECT'] = 'rlhf-reward-approx-v2'
os.environ["WANDB_CONSOLE"] = "wrap"

##=========Sweep Running & Training Code=========##
def run_sweep(config=None, sweep_config=None):
    with wandb.init(config=config) as run:
        wandb_config = wandb.config
        configuration = Configuration()
        serialized_config_id = configuration.set_configuration_hparams(wandb_config)
        output_dir = f"./run-files/{sweep_config['name']}/{run.name}"
        configuration.set_output_dir(output_dir)
        if not os.path.exists(configuration.OUTPUT_DIR + "/loggable"): os.makedirs(configuration.OUTPUT_DIR + "/loggable")
        with open(f'./{configuration.OUTPUT_DIR}/loggable/configuration.txt', 'w') as file:
            file.write(configuration.serialize())
            
        artifact = wandb.Artifact(name=f"sweep-files-{serialized_config_id}", type="configuration")
        
        model_kwargs = {
            'supervised-loss-weightage': configuration.SUPERVISED_LOSS_WEIGHTAGE,
            'value-head-dropout': configuration.VALUE_HEAD_DROPOUT,
            'model-update-every': configuration.OLD_MODEL_UPDATE_INTERVAL,
            'kl-penalty-mode': configuration.KL_PENALTY_MODE,
            'kl-beta': configuration.KL_BETA,
            'discount-factor': configuration.GAMMA,
            'gae-lambda': configuration.GAE_LAMBDA,
            'clip-lim-ppo-loss': configuration.CLIP_LIM
        }
        model = LimitedTrajectoryOpinionSummarizer(configuration.BACKBONE_NAME, configuration.TRAINING_MODE, configuration.RL_ALGORITHM, **model_kwargs)
        
        if configuration.MODEL_PRETRAINED_PATH is not None:
            print(f'Loading model checkpoint from {configuration.MODEL_PRETRAINED_PATH}')
            state_dict = torch.load(configuration.MODEL_PRETRAINED_PATH)
            model.load_state_dict(state_dict)
        
        train_dataset = ReviewsDataset(configuration.TRAIN_DATA_PATH, configuration.TRAINING_MODE, configuration.SCORING_MODE, 'train', configuration.SCORER_MODEL_KWARGS)
        valid_dataset = ReviewsDataset(configuration.VALID_DATA_PATH, configuration.TRAINING_MODE, configuration.SCORING_MODE, 'valid', configuration.SCORER_MODEL_KWARGS)
        amazon_test_dataset = ReviewsTestDataset(configuration.AMAZON_TEST_DATA_PATH)
        flipkart_test_dataset = ReviewsTestDataset(configuration.FLIPKART_TEST_DATA_PATH)
        oposum_test_dataset = ReviewsTestDataset(configuration.OPOSUM_TEST_DATA_PATH)
        
        training_args = TrainingArguments(
            output_dir=configuration.OUTPUT_DIR + "/ckpt",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=configuration.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=configuration.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=configuration.GRAD_ACC,
            num_train_epochs=configuration.EPOCHS,
            learning_rate=configuration.LEARNING_RATE,
            bf16=configuration.USE_BF16,
            fp16=configuration.USE_FP16,
            evaluation_strategy="steps",
            eval_steps=configuration.EVAL_STEPS,
            save_strategy="steps",
            save_steps=configuration.EVAL_STEPS,
            dataloader_num_workers=4,
            log_level="error",
            logging_strategy="steps",
            logging_steps=configuration.LOG_STEPS, 
            lr_scheduler_type=configuration.LR_SCHEDULER,
            warmup_steps=configuration.LR_WARMUP,
            optim=configuration.OPTIM_NAME,
            run_name=run.name,
            weight_decay=configuration.WEIGHT_DECAY,
            max_grad_norm=configuration.MAX_GRAD_NORM,
        )
        
        collator_fn = data_collator_fn_supervised
        if configuration.TRAINING_MODE == 'limited-trajectory-rl': collator_fn = data_collator_fn_limited_trajectory
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=lambda batch: collator_fn(batch, model.get_tokenizer(), model.get_max_length()),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            track_metric=configuration.TRACK_METRIC,
            goal=configuration.GOAL,
            amazon_test_dataset=amazon_test_dataset,
            flipkart_test_dataset=flipkart_test_dataset,
            oposum_test_dataset=oposum_test_dataset,
            output_dir_logging=configuration.OUTPUT_DIR + "/loggable"
        )
        
        callback = CustomTrainerCallback()
        trainer.add_callback(callback)
        trainer.add_output_dir(configuration.OUTPUT_DIR + "/loggable")
        trainer.add_generation_kwargs(configuration.GEN_KWARGS)
        
        summary = trainer.train()
        # trainer.save_model()
        
        artifact.add_dir(local_path=configuration.OUTPUT_DIR + "/loggable", name="train-artifacts")
        run.log_artifact(artifact)
        trainer.run_on_test_dataset(model.get_max_length(), configuration.OUTPUT_DIR + "/loggable")
        rmtree(configuration.OUTPUT_DIR)

if __name__ == '__main__':
    print(f'Starting training with {torch.cuda.device_count()} devices')
    sweep_id = wandb.sweep(SWEEP_CONFIGURATION, project='rlhf-reward-approx-v2')
    wandb.agent(sweep_id, lambda: run_sweep(sweep_config=SWEEP_CONFIGURATION), count=20)
