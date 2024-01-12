import torch
from torch.optim import Adam, AdamW
from model_code import LimitedTrajectoryOpinionSummarizer
from configuration import Configuration
from transformers import Trainer, TrainingArguments, TrainerCallback
from coolname import generate_slug
import wandb
import os
from data_handler import data_collator_fn_supervised, data_collator_fn_limited_trajectory, ReviewsDataset

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
        
##=========WandB Setup=========##
wandb.login()
os.environ['WANDB_PROJECT'] = 'hrl-options-qfs-sl-test-classif'
run_name = generate_slug(3)

if __name__ == '__main__':
    configuration = Configuration()
    configuration.set_run_name(run_name)
    model_kwargs = {'supervised-loss-weightage': configuration.SUPERVISED_LOSS_WEIGHTAGE}
    model = LimitedTrajectoryOpinionSummarizer(configuration.BACKBONE_NAME, configuration.TRAINING_MODE, **model_kwargs)
    train_dataset = ReviewsDataset(configuration.TRAIN_DATA_PATH, configuration.TRAINING_MODE)
    valid_dataset = ReviewsDataset(configuration.VALID_DATA_PATH, configuration.TRAINING_MODE)
    
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
        logging_steps=configuration.TRAIN_BATCH_SIZE, 
        lr_scheduler_type=configuration.LR_SCHEDULER,
        warmup_steps=configuration.LR_WARMUP,
        optim=configuration.OPTIMIZER_NAME,
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
        data_collator=lambda batch: collator_fn(batch, model.get_tokenizer()),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    callback = CustomTrainerCallback()
    trainer.add_callback(callback)
    
    summary = trainer.train()
    trainer.save_model()