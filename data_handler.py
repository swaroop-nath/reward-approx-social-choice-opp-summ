from torch.utils.data import Dataset, DataLoader
import json
import torch
from numpy.random import choice

class ReviewsDataset(Dataset):
    def __init__(self, data_path, training_mode):
        with open(data_path, 'r') as file:
            dataset = file.readlines()
        
        # A list of JSON objects {'reviews': list, 'summaries': [{'summary_text': str, 'score': float}], 'good': list, 'sbad': list, 'vbad': list}    
        self.dataset = self._process_dataset(dataset)
        self.size = len(self.dataset)
        self.training_mode = training_mode
        
    def _process_dataset(self, dataset):
        return [json.loads(line) for line in dataset]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
                
        reviews = item['reviews']
        review_text = ' '.join(reviews)
        
        good_summary_index = choice(item['good']) # Samples an index to select good summary
        good_summary = item['summaries'][good_summary_index]
        good_summary_text = good_summary['summary_text']
        
        if self.training_mode == 'supervised': return review_text, good_summary_text
        
        score_summary_index = choice(list(range(len(item['summaries'])))) # Samples an index from all the summaries
        score_summary = item['summaries'][score_summary_index]
        score_summary_text = score_summary['summary_text']
        score_summary_reward = score_summary['score']
        
        return review_text, good_summary_text, score_summary_text, score_summary_reward
    
def _tokenize_text(tokenizer, text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True) # Truncate to max length supported by the model
    
def data_collator_fn_supervised(batch, tokenizer):
    # batch is a list of 2-tuple
    review_texts, summaries = tuple(zip(*batch))
    review_texts = list(review_texts)
    summaries = list(summaries)
    
    review_texts_tokenized = _tokenize_text(tokenizer, review_texts)
    summaries_tokenized = _tokenize_text(tokenizer, summaries)
    
    return {
        'reviews-input-ids': review_texts_tokenized['input_ids'], # tensor
        'reviews-attention-mask': review_texts_tokenized['attention_mask'], # tensor
        'summaries-input-ids': summaries_tokenized['input_ids'][:, :-1], # tensor
        'summaries-attention-mask': summaries_tokenized['attention_mask'][:, :-1], # tensor
        'gt-summaries': summaries_tokenized['input_ids'][:, 1:], # tensor
    }
    
def data_collator_fn_limited_trajectory(batch, tokenizer):
    # batch is a list of 4-tuple
    reviews_texts, good_summaries, scoring_summaries, scoring_summary_rewards = tuple(zip(*batch))
    review_texts = list(review_texts)
    good_summaries = list(good_summaries)
    scoring_summaries  = list(scoring_summaries)
    scoring_summary_rewards = list(scoring_summary_rewards)
    
    review_texts_tokenized = _tokenize_text(tokenizer, review_texts)
    good_summaries_tokenized = _tokenize_text(tokenizer, good_summaries)
    scoring_summaries_tokenized = _tokenize_text(tokenizer, scoring_summaries)
    
    return {
        'sample-good': {
            'reviews-input-ids': review_texts_tokenized['input_ids'], # tensor
            'reviews-attention-mask': review_texts_tokenized['attention_mask'], # tensor
            'summaries-input-ids': good_summaries_tokenized['input_ids'][:, :-1], # tensor
            'summaries-attention-mask': good_summaries_tokenized['attention_mask'][:, :-1], # tensor
            'gt-summaries': good_summaries_tokenized['input_ids'][:, 1:], # tensor
        },
        'sample-scoring': {
            'reviews-input-ids': review_texts_tokenized['input_ids'], # tensor
            'reviews-attention-mask': review_texts_tokenized['attention_mask'], # tensor
            'summaries-input-ids': scoring_summaries_tokenized['input_ids'][:, :-1], # tensor
            'summaries-attention-mask': scoring_summaries_tokenized['attention_mask'][:, :-1], # tensor
            'output-summaries': scoring_summaries_tokenized['input_ids'][:, 1:], # tensor
            'rewards': torch.tensor(scoring_summary_rewards)
        }
    }
    