import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.distributions import Categorical

class LimitedTrajectoryOpinionSummarizer(nn.Module):
    def __init__(self, backbone_name, training_mode, **kwargs):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(backbone_name)
        self._tokenizer = BartTokenizerFast.from_pretrained(backbone_name)
        assert training_mode in ['supervised', 'limited-trajectory-rl'], f"{training_mode} not yet supported, please try one of `supervised` or `limited-trajectory-rl`"
        self.training_mode = training_mode
        if self.training_mode == 'limited-trajectory-rl':
            assert kwargs['supervised-loss-weightage'] >= 0, f"`supervised-loss-weightage` has to be atleast `0`"
            assert kwargs['supervised-loss-weightage'] < 1, f"`supervised-loss-weightage` can be `1` at most"
            self.supervised_loss_weightage = kwargs['supervised-loss-weightage']
            self.reinforcement_loss_weightage = 1 - kwargs['supervised-loss-weightage']
            
        self._log_dict = {}
            
    def get_tokenizer(self):
        return self._tokenizer
    
    def _update_logs(self, output_dict):
        for k, v in output_dict.items():
            output_dict[k] = v.unsqueeze(dim=0).reshape(-1)
        for k, v in output_dict.items():
            v = v.detach()
            existing_log = self._log_dict.get(k, None)
            if existing_log is None: self._log_dict[k] = v
            else: 
                new_log = torch.cat((existing_log, v.to(existing_log.device)), dim=-1).squeeze()
                self._log_dict[k] = new_log
    
    def get_logs(self):
        for k, v in self._log_dict.items():
            self._log_dict[k] = torch.mean(v).detach().item()
        return self._log_dict            
            
    def update_parameters_on_step_end(self):
        self._log_dict = {}
        
    def train_supervised(self, batch):
        model_output = self.model(input_ids=batch['reviews-input-ids'],
                                  attention_mask=batch['reviews-attention-mask'],
                                  decoder_input_ids=batch['summaries-input-ids'],
                                  decoder_attention_mask=batch['summaries-attention-mask'],
                                  labels=batch['gt-summaries'])
        
        cross_entropy_loss = model_output['loss']
        
        output_dict = {'loss': cross_entropy_loss, 'ce-loss': cross_entropy_loss}
        if self.training_mode == 'supervised': self._update_logs(output_dict)
        return output_dict
    
    def train_limited_trajectory_rl(self, batch):
        if self.supervised_loss_weightage > 0: 
            supervised_model_output = self.train_supervised(batch['sample-good'])
            cross_entropy_loss = supervised_model_output['ce-loss']
            
        model_output = self.model(input_ids=batch['sample-scoring']['reviews-input-ids'],
                                  attention_mask=batch['sample-scoring']['reviews-attention-mask'],
                                  decoder_input_ids=batch['sample-scoring']['summaries-input-ids'],
                                  decoder_attention_mask=batch['sample-scoring']['summaries-attention-mask'])
        
        # Reinforcement Learning based loss --> on a potentially not good/ideal [labelled] sample
        dist = Categorical(logits=model_output['logits'])
        log_probs = dist.log_prob(batch['sample-scoring']['output-summaries']) # (batch_size, seq_len)
        r_y_given_x = batch['summaries-scores'] # (batch_size,)
        
        # loss = - r(x, y) * π(y | x) | normalizing the log-prob by the length --> equivalent to a geometric mean of probabilities
        pi_y_given_x = (log_probs * batch['summaries-attention-mask']).sum(dim=1) / batch['summaries-attention-mask'].sum(dim=1)
        rl_loss = - torch.mean(r_y_given_x * pi_y_given_x)
        
        loss = self.supervised_loss_weightage * cross_entropy_loss + self.reinforcement_loss_weightage * rl_loss
        
        output_dict = {'loss': loss, 'ce-loss': cross_entropy_loss, 'rl-loss': rl_loss}
        self._update_logs(output_dict)
        return output_dict
        
    def forward(self, batch):
        if self.training_mode == 'supervised': return self.train_supervised(batch)
        elif self.training_mode == 'limited-trajectory-rl': return self.train_limited_trajectory_rl(batch)
        