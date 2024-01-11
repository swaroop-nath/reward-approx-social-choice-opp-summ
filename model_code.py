import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from torch.distributions import Categorical

class LimitedTrajectoryOpinionSummarizer(nn.Module):
    def __init__(self, backbone_name, training_mode, **kwargs):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(backbone_name)
        assert training_mode in ['supervised', 'limited-trajectory-rl'], f"{training_mode} not yet supported, please try one of `supervised` or `limited-trajectory-rl`"
        self.training_mode = training_mode
        if self.training_mode == 'limited-trajectory-rl':
            assert kwargs['supervised-loss-weightage'] >= 0, f"`supervised-loss-weightage` has to be atleast `0`"
            assert kwargs['supervised-loss-weightage'] < 1, f"`supervised-loss-weightage` can be `1` at most"
            self.supervised_loss_weightage = kwargs['supervised-loss-weightage']
            self.reinforcement_loss_weightage = 1 - kwargs['supervised-loss-weightage']
        
    def train_supervised(self, batch):
        model_output = self.model(input_ids=batch['reviews-input-ids'],
                                  attention_mask=batch['reviews-attention-mask'],
                                  decoder_input_ids=batch['summaries-input-ids'],
                                  decoder_attention_mask=batch['summaries-attention-mask'],
                                  labels=batch['gt-summaries'])
        
        cross_entropy_loss = model_output['loss']
        
        return {'loss': cross_entropy_loss, 'ce-loss': cross_entropy_loss}
    
    def train_limited_trajectory_rl(self, batch):
        model_output = self.model(input_ids=batch['reviews-input-ids'],
                                  attention_mask=batch['reviews-attention-mask'],
                                  decoder_input_ids=batch['summaries-input-ids'],
                                  decoder_attention_mask=batch['summaries-attention-mask'],
                                  labels=batch['gt-summaries'])
        
        cross_entropy_loss = model_output['loss']
        
        # Reinforcement Learning based loss
        dist = Categorical(logits=model_output['logits'])
        log_probs = dist.log_prob(batch['gt-summaries']) # (batch_size, seq_len)
        r_y_given_x = batch['summaries-scores'] # (batch_size,)
        
        # loss = - r(x, y) * π(y | x)
        pi_y_given_x = (log_probs * batch['summaries-attention-mask']).sum(dim=1) / batch['summaries-attention-mask'].sum(dim=1)
        rl_loss = - torch.mean(r_y_given_x * pi_y_given_x)
        
        loss = self.supervised_loss_weightage * cross_entropy_loss + self.reinforcement_loss_weightage * rl_loss
        
        return {'loss': loss, 'ce-loss': cross_entropy_loss, 'rl-loss': rl_loss}
        
    def forward(self, batch):
        if self.training_mode == 'supervised': return self.train_supervised(batch)
        elif self.training_mode == 'limited-trajectory-rl': return self.train_limited_trajectory_rl(batch)
        