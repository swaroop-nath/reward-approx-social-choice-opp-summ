import torch
import torch.nn as nn
import os
import wandb
from model_code import FFRewardModel
from data_handler import SyntheticFeedbackDataset, data_collator_fn_for_synthetic_feedback
from configuration import Configuration, SWEEP_CONFIGURATION
from transformers import TrainingArguments, Trainer
from shutil import rmtree

##=========WandB Setup=========##
wandb.login()
os.environ['WANDB_PROJECT'] = 'rlhf-reward-approx-rm-train'
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
        
        model = FFRewardModel(configuration.LAYER_CONFIG, configuration.NUM_INPUTS, configuration.NUM_OUTPUTS, configuration.ACTIVATION)
        if configuration.RM_TRAIN == 'synthetic-feedback':
            train_dataset = SyntheticFeedbackDataset(configuration.TRAIN_DATA_PATH, configuration.TRAIN_SAMPLE_SIZE)
            valid_dataset = SyntheticFeedbackDataset(configuration.VALID_DATA_PATH, configuration.VALID_SAMPLE_SIZE)
            collator_fn = data_collator_fn_for_synthetic_feedback
        elif configuration.RM_TRAIN == 'human-feedback': raise NotImplementedError
        
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

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator_fn,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )
        
        trainer.train()
        trainer.save_model()
        artifact.add_dir(local_path=configuration.OUTPUT_DIR + "/loggable", name="train-artifacts")
        run.log_artifact(artifact)
        rmtree(configuration.OUTPUT_DIR)
        
if __name__ == '__main__':
    sweep_id = wandb.sweep(SWEEP_CONFIGURATION, project='rlhf-reward-approx-rm-train')
    wandb.agent(sweep_id, lambda: run_sweep(sweep_config=SWEEP_CONFIGURATION), count=20)