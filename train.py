from model_code import LimitedTrajectoryOpinionSummarizer
from configuration import Configuration
from transformers import Trainer, TrainingArguments, TrainerCallback
from coolname import generate_slug
import wandb
import os
from tqdm import tqdm
import json
from utils import get_rouge_score
import torch
from data_handler import data_collator_fn_supervised, data_collator_fn_limited_trajectory, ReviewsDataset, ReviewsTestDataset

##=========Custom Callback for Logging Setup=========##
class CustomTrainerCallback(TrainerCallback):
    def add_model(self, model):
        self._model = model
        
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
        
##=========Custom Trainer for Logging Setup=========##
class CustomTrainer(Trainer):
    def add_callback(self, callback):
        super().add_callback(callback)
        if isinstance(callback, CustomTrainerCallback): callback.add_model(self.model)
        
    def add_run_name(self, run_name):
        self.run_name = run_name
        
    def add_generation_kwargs(self, generation_kwargs):
        self.generation_kwargs = generation_kwargs
        
    def run_on_test_dataset(self, test_dataset, max_length):
        tokenizer = self.model.get_tokenizer()
        
        output_dir = f"test-output-dir/{self.run_name}"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        pbar = tqdm(total=len(test_dataset), desc='Running on test set')
        writeable = []
        true_summaries, pred_summaries = [], []
        for unique_id, review_text, summary in test_dataset:
            review_input_ids = tokenizer([review_text], return_tensors='pt')['input_ids']
            batch = {'input_ids': review_input_ids[:, :max_length]}
            batch = self._prepare_inputs(batch)
            output = self.model.generate(batch['input_ids'], **generation_kwargs)[0]
            gen_summary = tokenizer.decode(output)
            
            true_summaries.append(summary)
            pred_summaries.append(gen_summary)
            
            writeable.append(json.dumps({'unique-id': unique_id, 'review-text': review_text, 'gt-summary': summary, 'pred-summary': gen_summary}))
            pbar.update(1)
            
        pbar.close()
        
        scores = get_rouge_score(pred_summaries, true_summaries)
        
        fname = '-'.join([str(key) + '-' + str(val) for key, val in generation_kwargs.items()])
        with open(f"{output_dir}/{fname}.jsonl", 'w') as file:
            file.write('\n'.join(writeable))
            
        with open(f"{output_dir}/scores.json", 'w') as file:
            json.dump(scores, file)
        
##=========WandB Setup=========##
wandb.login()
os.environ['WANDB_PROJECT'] = 'rlhf-reward-approx'
run_name = generate_slug(3)

if __name__ == '__main__':
    configuration = Configuration()
    configuration.set_run_name(run_name)
    model_kwargs = {'supervised-loss-weightage': configuration.SUPERVISED_LOSS_WEIGHTAGE}
    model = LimitedTrajectoryOpinionSummarizer(configuration.BACKBONE_NAME, configuration.TRAINING_MODE, **model_kwargs)
    
    if configuration.MODEL_PRETRAINED_PATH is not None:
        print(f'Loading model checkpoint from {configuration.MODEL_PRETRAINED_PATH}')
        state_dict = torch.load(configuration.MODEL_PRETRAINED_PATH)
        model.load_state_dict(state_dict)
    
    train_dataset = ReviewsDataset(configuration.TRAIN_DATA_PATH, configuration.TRAINING_MODE, configuration.SCORING_MODE)
    valid_dataset = ReviewsDataset(configuration.VALID_DATA_PATH, configuration.TRAINING_MODE, configuration.SCORING_MODE)
    test_dataset = ReviewsTestDataset(configuration.TEST_DATA_PATH)
    
    generation_kwargs = configuration.GEN_KWARGS
    
    training_args = TrainingArguments(
        output_dir=configuration.OUTPUT_DIR,
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
        run_name=configuration.RUN_NAME,
        weight_decay=configuration.WEIGHT_DECAY,
        max_grad_norm=configuration.MAX_GRAD_NORM,
        report_to='wandb'
    )
    
    collator_fn = data_collator_fn_supervised
    if configuration.TRAINING_MODE == 'limited-trajectory-rl': collator_fn = data_collator_fn_limited_trajectory
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=lambda batch: collator_fn(batch, model.get_tokenizer(), model.get_max_length()),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    callback = CustomTrainerCallback()
    trainer.add_callback(callback)
    trainer.add_run_name(run_name)
    
    # trainer.run_on_test_dataset(test_dataset, model.get_max_length())
    summary = trainer.train()
    trainer.save_model()
    trainer.run_on_test_dataset(test_dataset, model.get_max_length())