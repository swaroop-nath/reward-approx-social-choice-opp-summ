from torch.utils.data import Dataset, DataLoader
import json
import torch
from numpy.random import choice
from pandas import read_csv
import numpy as np
from .reward_modelling.model_code import FFRewardModel

class ReviewsTestDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = read_csv(data_path)
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        
        unique_id = row['group_id']
        review_text = '\n'.join(eval(row['review_text']))
        summary = eval(row['summary'])[0]
        
        return unique_id, review_text, summary

class ReviewsDataset(Dataset):
    def __init__(self, data_path, training_mode, scoring_mode, split, scorer_model_kwargs=None):
        with open(data_path, 'r') as file:
            dataset = file.readlines()
        
        # A list of JSON objects {'unique-id': uuid, 'reviews': list, 'summaries': [{'summary_text': str, 'score': float, 'is-good': bool, 'is-sbad': bool, 'is-vbad': bool}]}
        self.dataset = self._process_dataset(dataset)
        self.size = len(self.dataset)
        self.training_mode = training_mode
        assert scoring_mode in ['naive-mean', 'synthetic-feedback', 'human-feedback'], f"Specified scoring-mode {scoring_mode} is not supported yet"
        self.scoring_mode = scoring_mode
        self.split = split
        if self.scoring_mode == 'synthetic-feedback':
            self._scorer_model = FFRewardModel(scorer_model_kwargs['layer-config'], scorer_model_kwargs['num-inputs'], scorer_model_kwargs['num-outputs'], scorer_model_kwargs['activation'])
            self._scorer_model.load_state_dict(torch.load(scorer_model_kwargs['pretrained-path'], map_location='cpu'))
        
        self._reward_metrics = set(['aspect-coverage', 'opinion-faithfulness', 'opinion-coverage', 'conciseness', 'relevance', 'hallucination', 'language-correctness'])
        
    def _process_dataset(self, dataset):
        return [json.loads(line) for line in dataset]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
                
        unique_id = item['unique-id']
        reviews = item['reviews']
        review_text = ' '.join(reviews)
        
        summaries = item['summaries']
        key = 'is-sbad' if self.split == 'train' else 'is-good'
        gt_summaries = [summary['summary_text'] for summary in summaries if summary[key]]
        
        if self.training_mode == 'supervised':
            return unique_id, review_text, gt_summaries[0]
        
        gt_summary_index = choice(range(len(gt_summaries)))
        gt_summary_text = gt_summaries[gt_summary_index]
            
        score_summary_index = choice(range(len(item['summaries']))) # Samples an index from all the summaries
        score_summary = item['summaries'][score_summary_index]
        score_summary_reward = self._get_aggregate_reward(score_summary['score'])
        counter = 0
        while np.isnan(score_summary_reward) or len(score_summary['summary_text']) == 0: 
            score_summary_index = choice(range(len(item['summaries'])))
            score_summary = item['summaries'][score_summary_index]
            counter += 1
            score_summary_reward = self._get_aggregate_reward(score_summary['score'])
            
            if counter > 10: raise RuntimeError(f"nan encountered more than 10 times for unique-id: {unique_id}")
            
        score_summary_text = score_summary['summary_text']
        
        return unique_id, review_text, gt_summary_text, score_summary_text, score_summary_reward
    
    def _get_aggregate_reward(self, scores):
        if self.scoring_mode == 'naive-mean': 
            score_vals = [v for k, v in scores.items() if k in self._reward_metrics]
            return np.nanmean(score_vals) / 5.0 # Normalizing to put the score between 0 and 1
        elif self.scoring_mode in ['synthetic-feedback', 'human-feedback']:
            score_vals = [v for k, v in scores.items() if k in self._reward_metrics]
            X = torch.tensor(score_vals).unsqueeze(dim=0) # (1, num_input/num_reward_metrics)
            reward = self._scorer_model.get_reward(X).squeeze().item()
            return reward
        else: raise NotImplementedError(f"scoring-mode {self.scoring_mode} is not yet implemented.")
    
def _tokenize_text(tokenizer, text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True) # Truncate to max length supported by the model
    
def data_collator_fn_supervised(batch, tokenizer, max_length=1024):
    # batch is a list of 2-tuple
    unique_ids, review_texts, summaries = tuple(zip(*batch))
    unique_ids = list(unique_ids)
    review_texts = list(review_texts)
    summaries = list(summaries)
    
    review_texts_tokenized = _tokenize_text(tokenizer, review_texts)
    summaries_tokenized = _tokenize_text(tokenizer, summaries)
    
    return {
        'unique-ids': unique_ids,
        'reviews-input-ids': review_texts_tokenized['input_ids'][:, :max_length], # tensor
        'reviews-attention-mask': review_texts_tokenized['attention_mask'][:, :max_length], # tensor
        'summaries-input-ids': summaries_tokenized['input_ids'][:, :max_length][:, :-1], # tensor
        'summaries-attention-mask': summaries_tokenized['attention_mask'][:, :max_length][:, :-1], # tensor
        'gt-summaries': summaries_tokenized['input_ids'][:, :max_length][:, 1:], # tensor
    }
    
def data_collator_fn_limited_trajectory(batch, tokenizer, max_length=1024):
    # batch is a list of 4-tuple
    unique_ids, review_texts, good_summaries, scoring_summaries, scoring_summary_rewards = tuple(zip(*batch))
    unique_ids = list(unique_ids)
    review_texts = list(review_texts)
    good_summaries = list(good_summaries)
    scoring_summaries  = list(scoring_summaries)
    scoring_summary_rewards = list(scoring_summary_rewards)
    
    review_texts_tokenized = _tokenize_text(tokenizer, review_texts)
    good_summaries_tokenized = _tokenize_text(tokenizer, good_summaries)
    scoring_summaries_tokenized = _tokenize_text(tokenizer, scoring_summaries)
    
    return {
        'sample-good': {
            'unique-ids': unique_ids,
            'reviews-input-ids': review_texts_tokenized['input_ids'][:, :max_length], # tensor
            'reviews-attention-mask': review_texts_tokenized['attention_mask'][:, :max_length], # tensor
            'summaries-input-ids': good_summaries_tokenized['input_ids'][:, :max_length][:, :-1], # tensor
            'summaries-attention-mask': good_summaries_tokenized['attention_mask'][:, :max_length][:, :-1], # tensor
            'gt-summaries': good_summaries_tokenized['input_ids'][:, :max_length][:, 1:], # tensor
        },
        'sample-scoring': {
            'unique-ids': unique_ids,
            'reviews-input-ids': review_texts_tokenized['input_ids'][:, :max_length], # tensor
            'reviews-attention-mask': review_texts_tokenized['attention_mask'][:, :max_length], # tensor
            'summaries-input-ids': scoring_summaries_tokenized['input_ids'][:, :max_length][:, :-1], # tensor
            'summaries-attention-mask': scoring_summaries_tokenized['attention_mask'][:, :max_length][:, :-1], # tensor
            'output-summaries': scoring_summaries_tokenized['input_ids'][:, :max_length][:, 1:], # tensor
            'rewards': torch.tensor(scoring_summary_rewards)
        }
    }
    