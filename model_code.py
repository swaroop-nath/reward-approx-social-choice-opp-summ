import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.distributions import Categorical

class ValueHead(nn.Module):
    def __init__(self, input_size, dropout_prob):
        super().__init__()
        self.embedding_dropout = nn.Dropout(p=dropout_prob)
        self.value_head = nn.Linear(in_features=input_size, out_features=1)
        
    def forward(self, input_hidden_states):
        input_hidden_states = self.embedding_dropout(input_hidden_states)
        return self.value_head(input_hidden_states).squeeze(dim=-1) # (bsz, seq_len)

class LimitedTrajectoryOpinionSummarizer(nn.Module):
    def __init__(self, backbone_name, training_mode, rl_algorithm, **kwargs):
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
            
        assert rl_algorithm in ['policy-gradient', 'proximal-policy-optimization']
        self._rl_algorithm = rl_algorithm
        self._epsilon = torch.tensor(1e-9)
        
        if self._rl_algorithm == 'proximal-policy-optimization': 
            self._value_head = ValueHead(self.model.config.hidden_size, kwargs['value-head-dropout'])
            self._old_model_update_counter = 0
            self._old_model_update_interval = kwargs['model-update-every']
            self._old_model = BartForConditionalGeneration.from_pretrained(backbone_name)
            self._kl_penalty_mode = kwargs['kl-penalty-mode']
            assert self._kl_penalty_mode in ['instruct-gpt', 'abs', 'square'], f"Specified kl-penalty-mode `{self._kl_penalty_mode}` not implemented"
            self._kl_beta = kwargs['kl-beta']
            self._gamma = kwargs['discount-factor']
            self._gae_lambda = kwargs['gae-lambda']
            self._clip_lim = kwargs['clip-lim-ppo-loss']
            
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
    
    def get_max_length(self):
        return self.model.config.max_position_embeddings
    
    def get_logs(self):
        for k, v in self._log_dict.items():
            self._log_dict[k] = torch.mean(v).detach().item()
        return self._log_dict         
            
    def update_parameters_on_step_end(self):
        self._log_dict = {}
      
    @torch.no_grad()  
    def generate(self, batch, **generation_kwargs):
        return self.model.generate(**batch, **generation_kwargs)
        
    def train_supervised(self, **batch):
        model_output = self.model(input_ids=batch['reviews-input-ids'],
                                  attention_mask=batch['reviews-attention-mask'],
                                  decoder_input_ids=batch['summaries-input-ids'],
                                  decoder_attention_mask=batch['summaries-attention-mask'],
                                  labels=batch['gt-summaries'])
        
        cross_entropy_loss = model_output['loss']
        
        output_dict = {'loss': cross_entropy_loss, 'ce-loss': cross_entropy_loss}
        if self.training_mode == 'supervised': self._update_logs(output_dict) # --> TODO: This detaches loss!! DONE
        return {'loss': cross_entropy_loss, 'ce-loss': cross_entropy_loss}
    
    def train_limited_trajectory_rl_policy_gradient(self, **batch):
        cross_entropy_loss = None
        if self.supervised_loss_weightage > 0: 
            supervised_model_output = self.train_supervised(**batch['sample-good'])
            cross_entropy_loss = supervised_model_output['ce-loss']
            
        model_output = self.model(input_ids=batch['sample-scoring']['reviews-input-ids'],
                                  attention_mask=batch['sample-scoring']['reviews-attention-mask'],
                                  decoder_input_ids=batch['sample-scoring']['summaries-input-ids'],
                                  decoder_attention_mask=batch['sample-scoring']['summaries-attention-mask'])
        
        # Reinforcement Learning based loss --> on a potentially not good/ideal [labelled] sample
        dist = Categorical(logits=model_output['logits'])
        log_probs = dist.log_prob(batch['sample-scoring']['output-summaries']) # (batch_size, seq_len)
        r_y_given_x = batch['sample-scoring']['rewards'] # (batch_size,)
        
        # loss = - r(x, y) * π(y | x) | normalizing the log-prob by the length --> equivalent to a geometric mean of probabilities
        pi_y_given_x = (log_probs * batch['sample-scoring']['summaries-attention-mask']).sum(dim=1) / batch['sample-scoring']['summaries-attention-mask'].sum(dim=1)
        rl_loss = - torch.mean(r_y_given_x * pi_y_given_x)
        
        loss = self.reinforcement_loss_weightage * rl_loss
        output_dict = {'loss': loss, 'rl-loss': rl_loss, 'pi_y_given_x': torch.mean(pi_y_given_x), 'reward': torch.mean(r_y_given_x)}
        if cross_entropy_loss is not None: 
            loss = loss + self.supervised_loss_weightage * cross_entropy_loss 
            output_dict.update({'ce-loss': cross_entropy_loss})
        
        self._update_logs(output_dict) # --> TODO: This detaches loss!! DONE
        if cross_entropy_loss is None: return {'loss': loss, 'rl-loss': rl_loss}
        return {'loss': loss, 'ce-loss': cross_entropy_loss, 'rl-loss': rl_loss}
    
    def train_limited_trajectory_rl_ppo(self, **batch):
        model_output = self.model(input_ids=batch['sample-scoring']['reviews-input-ids'],
                                  attention_mask=batch['sample-scoring']['reviews-attention-mask'],
                                  decoder_input_ids=batch['sample-scoring']['summaries-input-ids'],
                                  decoder_attention_mask=batch['sample-scoring']['summaries-attention-mask'],
                                  output_hidden_states=True)
        
        old_log_probs, old_values = self.get_ref_outputs(**batch)
        old_log_probs = old_log_probs * batch['sample-scoring']['summaries-attention-mask'] # Non-zero only at unmasked places
        
        # Reinforcement Learning based loss --> on a potentially not good/ideal [labelled] sample
        dist = Categorical(logits=model_output['logits'])
        log_probs = dist.log_prob(batch['sample-scoring']['output-summaries']) # (batch_size, seq_len)
        log_probs = log_probs * batch['sample-scoring']['summaries-attention-mask'] # Non-zero only at unmasked places
        scores = batch['sample-scoring']['rewards'] # (batch_size,)
        
        values = self._value_head(model_output['decoder_hidden_states'][-1]) # (batch_size, seq_len)
        values = values * batch['sample-scoring']['summaries-attention-mask'] # Non-zero only at unmasked locations
        rewards = self._compute_rewards(scores, log_probs, old_log_probs, batch['sample-scoring']['summaries-attention-mask'])
        gae, returns = self._compute_advantage_and_return(values, rewards, batch['sample-scoring']['summaries-attention-mask'])
        pg_loss, vf_loss = self.ppo_loss(old_log_probs, old_values, log_probs, values, gae, returns, batch['sample-scoring']['summaries-attention-mask'])
        
        # Updating ref model
        self._old_model_update_counter = (self._old_model_update_counter + 1) % self._old_model_update_interval
        if self._old_model_update_counter == self._old_model_update_interval - 1:
            self._old_model.load_state_dict(self.model.state_dict())
        
        output_dict = {'pg-loss': pg_loss, 'vf-loss': vf_loss, 'loss': pg_loss + vf_loss}
        self._update_logs(output_dict)
        return {'pg-loss': pg_loss, 'vf-loss': vf_loss, 'loss': pg_loss + vf_loss}
      
    def ppo_loss(self, old_log_probs, old_values, log_probs, values, gae, returns, attn_mask):
        # Value Function Loss
        vmin = old_values - self._clip_lim
        vmax = old_values + self._clip_lim
        values_clipped = torch.max(vmin, torch.min(values, vmax))
        
        vf_loss_1 = 0.5 * (values - returns).square() # values back-props the gradients, returns in non-differentiable
        vf_loss_2 = 0.5 * (values_clipped - returns).square()
        
        vf_loss = torch.max(vf_loss_1, vf_loss_2) # Clipping
        vf_loss_mean = torch.mean(torch.sum(vf_loss, dim=1) / torch.sum(attn_mask, dim=1)) # scalar
        
        # Policy Loss
        ratio = torch.exp(log_probs - old_log_probs)
        rmin = torch.tensor(1.0 - self._clip_lim, dtype=ratio.dtype, device=ratio.device)
        rmax = torch.tensor(1.0 + self._clip_lim, dtype=ratio.dtype, device=ratio.device)
        ratio_clipped = torch.max(rmin, torch.min(rmax, ratio))
        pg_loss_1 = gae * ratio
        pg_loss_2 = gae * ratio_clipped
        pg_loss = - torch.min(pg_loss_1, pg_loss_2) # Clipping
        pg_loss_mean = torch.mean(torch.sum(pg_loss, dim=1) / torch.sum(attn_mask, dim=1)) # scalar
        
        return pg_loss_mean, vf_loss_mean
        
    def _compute_advantage_and_return(self, values, rewards, attn_mask):
        # values.size() == rewards.size() == attn_mask.size() == (batch_size, seq_len)
        values_next = torch.cat((values[:, 1:], torch.zeros((values.size(0), 1), device=values.device, dtype=values.dtype)), dim=1)
        deltas = rewards + self._gamma * values_next - values # r(t) + γ * V(t + 1) - V(t)
        
        gae_weight_matrix = attn_mask * self._gamma * self._gae_lambda # γ * λ only at non-masked regions, else 1
        gae_weight_matrix = torch.cumprod(gae_weight_matrix, dim=1) / (self._gamma * self._gae_lambda) # [1, γλ, (γλ)^2, . . .] 
        gae_weight_matrix = attn_mask * gae_weight_matrix + (1 - attn_mask) * gae_weight_matrix
        
        gae_bare = gae_weight_matrix * deltas
        gae = torch.flip(torch.cumsum(torch.flip(gae_bare, dims=(1,)), dim=1), dims=(1,)) / torch.max(gae_weight_matrix, self._epsilon.to(gae_weight_matrix.device)) # At = δt + γλ*δ{t+1} + γλ*δ{t+2} + . . .
        
        returns = gae + values_next
        return gae.detach(), returns.detach() # Only log-probs and values are supposed to be differentiable | Both zero at unmasked places
        
    def _compute_rewards(self, scores, log_probs, old_log_probs, attn_mask):
        # scores.size() == (batch_size,)
        # log_probs.size() == old_log_probs.size() == attn_mask.size() == (batch_size, seq_len)
        kl_penalty = self._compute_kl_penalty(log_probs, old_log_probs)
        monotonically_increasing_seq = torch.arange(1, kl_penalty.size(1) + 1, device=kl_penalty.device)
        last_non_zero_indices = torch.argmax(attn_mask * monotonically_increasing_seq, dim=-1)
        
        rewards = - kl_penalty * self._kl_beta
        rewards[torch.arange(scores.size(0)), last_non_zero_indices] += scores
        
        return rewards * attn_mask # Non-zero only at unmasked places | (batch_size, seq_len)
        
    def _compute_kl_penalty(self, log_probs, old_log_probs):
        if self._kl_penalty_mode == 'instruct-gpt':
            return log_probs - old_log_probs
        elif self._kl_penalty_mode == 'abs':
            return (log_probs - old_log_probs).abs()
        elif self._kl_penalty_mode == 'square':
            return 0.5 * (log_probs - old_log_probs).square()
        
    @torch.no_grad()
    def get_ref_outputs(self, **batch):
        old_model_output = self._old_model(input_ids=batch['sample-scoring']['reviews-input-ids'],
                                  attention_mask=batch['sample-scoring']['reviews-attention-mask'],
                                  decoder_input_ids=batch['sample-scoring']['summaries-input-ids'],
                                  decoder_attention_mask=batch['sample-scoring']['summaries-attention-mask'],
                                  output_hidden_states=True)
        
        old_values = self._value_head(old_model_output['decoder_hidden_states'][-1])
        old_dist = Categorical(logits=old_model_output['logits'])
        old_log_probs = old_dist.log_prob(batch['sample-scoring']['output-summaries'])
        
        return old_log_probs, old_values
    
    def train_limited_trajectory_rl(self, **batch):
        if self._rl_algorithm == 'policy-gradient': return self.train_limited_trajectory_rl_policy_gradient(**batch)
        elif self._rl_algorithm == 'proximal-policy-optimization': return self.train_limited_trajectory_rl_ppo(**batch)
        
    def forward(self, **batch):
        if self.training_mode == 'supervised': return self.train_supervised(**batch)
        elif self.training_mode == 'limited-trajectory-rl': return self.train_limited_trajectory_rl(**batch)
        