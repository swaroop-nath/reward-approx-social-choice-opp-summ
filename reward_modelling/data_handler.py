from torch.utils.data import Dataset
import json
import numpy as np
from pandas import DataFrame
import torch

class SyntheticFeedbackDataset(Dataset):
    def __init__(self, data_path, sample_size):
        with open(data_path) as file:
            lines = file.readlines()
        self._dataset = [json.loads(line) for line in lines]
        self._samples = np.random.choice(range(len(self._dataset)), size=sample_size, replace=False)
        self._reward_metrics = set(['aspect-coverage', 'opinion-faithfulness', 'opinion-coverage', 'conciseness', 'relevance', 'hallucination', 'language-correctness'])
        
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, idx):
        item_idx = self._samples[idx]
        data_item = self._dataset[item_idx]
        
        unique_id = data_item['unique-id']
        summaries = data_item['summaries']
        scores_categorized = {
            'is-good': [summary['score'] for summary in summaries if summary['is-good']],
            'is-sbad': [summary['score'] for summary in summaries if summary['is-sbad']],
            'is-vbad': [summary['score'] for summary in summaries if summary['is-vbad']]
        }
        
        comparable = np.random.choice(['is-good', 'is-sbad', 'is-vbad'], size=2, replace=False)
        iter_idx = 0
        while len(scores_categorized[comparable[0]]) == 0 or len(scores_categorized[comparable[1]]) == 0:
            comparable = np.random.choice(['is-good', 'is-sbad', 'is-vbad'], size=2, replace=False)
            iter_idx = 0
            if iter_idx > 0: raise RuntimeError(f"Iterated over 10 times with an empty comparable set | Unique ID: {unique_id}")
        comparable = self._sort_preferred(comparable) # 0th is the preferred one
        
        comparable_one_index = np.random.choice(len(scores_categorized[comparable[0]]), size=1)[0]
        comparable_two_index = np.random.choice(len(scores_categorized[comparable[1]]), size=1)[0]
        
        scores_preferred = scores_categorized[comparable[0]][comparable_one_index]
        scores_unpreferred = scores_categorized[comparable[1]][comparable_two_index]
        
        scores_preferred_metrics = {k: v for k, v in scores_preferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        scores_unpreferred_metrics = {k: v for k, v in scores_unpreferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        
        return scores_preferred_metrics, scores_unpreferred_metrics
        
    def _sort_preferred(self, comparable):
        if comparable[0] == 'is-good': return comparable # ['is-good', *]
        if comparable[0] == 'is-sbad':
            if comparable[1] == 'is-vbad': return comparable # ['is-sbad', 'is-vbad']
            return [comparable[1], comparable[0]] # ['is-sbad', 'is-good'] --> ['is-good', 'is-sbad']
        if comparable[0] == 'is-vbad': return [comparable[1], comparable[0]] # ['is-vbad', *] --> [*, 'is-vbad']
        
class HumanFeedbackDataset(Dataset):
    def __init__(self, data_path, train_size, use_aux=False, aux_data_path=None, aux_data_frac=0.25, mode='train'):
        self._data_path = data_path
        with open(data_path) as file:
            lines = file.readlines()
        self._dataset = [json.loads(line) for line in lines]
        if train_size != len(self._dataset):
            smaller_dataset = np.random.choice(self._dataset, train_size, replace=False)
            self._dataset = smaller_dataset
            print(f"Training with dataset with samples: {len(self._dataset)}")
        self._reward_metrics = set(['aspect-coverage', 'opinion-faithfulness', 'opinion-coverage', 'conciseness', 'relevance', 'hallucination', 'language-correctness'])
        
        self._mode = mode
        self._use_aux = use_aux
        
        if self._mode == 'train' and self._use_aux:
            with open(aux_data_path) as file:
                lines = file.readlines()
            self._aux_dataset = [json.loads(line) for line in lines]
            self._aux_data_frac = aux_data_frac
        
    def __len__(self):
        if self._use_aux: return int(len(self._dataset) * (1 + self._aux_data_frac))
        return len(self._dataset)
    
    def _get_item_aux(self, idx):
        data_item = self._aux_dataset[idx]
        
        unique_id = data_item['unique-id']
        summaries = data_item['summaries']
        scores_categorized = {
            'is-good': [summary['score'] for summary in summaries if summary['is-good']],
            'is-sbad': [summary['score'] for summary in summaries if summary['is-sbad']],
            'is-vbad': [summary['score'] for summary in summaries if summary['is-vbad']]
        }
        
        comparable = np.random.choice(['is-good', 'is-sbad', 'is-vbad'], size=2, replace=False)
        iter_idx = 0
        while len(scores_categorized[comparable[0]]) == 0 or len(scores_categorized[comparable[1]]) == 0:
            comparable = np.random.choice(['is-good', 'is-sbad', 'is-vbad'], size=2, replace=False)
            iter_idx = 0
            if iter_idx > 0: raise RuntimeError(f"Iterated over 10 times with an empty comparable set | Unique ID: {unique_id}")
        comparable = self._sort_preferred(comparable) # 0th is the preferred one
        
        comparable_one_index = np.random.choice(len(scores_categorized[comparable[0]]), size=1)[0]
        comparable_two_index = np.random.choice(len(scores_categorized[comparable[1]]), size=1)[0]
        
        scores_preferred = scores_categorized[comparable[0]][comparable_one_index]
        scores_unpreferred = scores_categorized[comparable[1]][comparable_two_index]
        
        scores_preferred_metrics = {k: v for k, v in scores_preferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        scores_unpreferred_metrics = {k: v for k, v in scores_unpreferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        
        return scores_preferred_metrics, scores_unpreferred_metrics
    
    def _sort_preferred(self, comparable):
        if comparable[0] == 'is-good': return comparable # ['is-good', *]
        if comparable[0] == 'is-sbad':
            if comparable[1] == 'is-vbad': return comparable # ['is-sbad', 'is-vbad']
            return [comparable[1], comparable[0]] # ['is-sbad', 'is-good'] --> ['is-good', 'is-sbad']
        if comparable[0] == 'is-vbad': return [comparable[1], comparable[0]] # ['is-vbad', *] --> [*, 'is-vbad']
    
    def __getitem__(self, idx):
        if self._mode == 'train' and self._use_aux and np.random.rand() <= self._aux_data_frac:
            return self._get_item_aux(idx)
        data_item = self._dataset[idx]
        summary_pairs = data_item['summary-pairs']
        assert data_item['preference-data'] in [1, 2], f"Preference data `{data_item['preference-data']}` invalid"
        preferred = 'summary-one' if data_item['preference-data'] == 1 else 'summary-two'
        unpreferred = 'summary-one' if data_item['preference-data'] == 2 else 'summary-two'
        
        summary_preferred = summary_pairs[preferred]
        summary_unpreferred = summary_pairs[unpreferred]
        scores_preferred = summary_preferred['summary-score']
        scores_unpreferred = summary_unpreferred['summary-score']
        
        scores_preferred_metrics = {k: v for k, v in scores_preferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        scores_unpreferred_metrics = {k: v for k, v in scores_unpreferred.items() if k in self._reward_metrics} # {metric: val} for metric in the 7 metrics
        
        return scores_preferred_metrics, scores_unpreferred_metrics
     
def _get_batched(scores, return_tensors='pt'):
    df = DataFrame.from_records(scores)
    batch = [df.iloc[i].tolist() for i in range(df.shape[0])]
    if return_tensors == 'np': return np.array(batch) / 5.0 # Normalized to 0 -- 1 range
    elif return_tensors == 'pt': return torch.tensor(batch) / 5.0 # Normalized to 0 -- 1 range
        
def data_collator_fn_for_synthetic_feedback(batch):
    scores_preferred_metrics, scores_unpreferred_metrics = tuple(zip(*batch))
    # scores_preferred_metrics, scores_unpreferred_metrics --> list/tuple of dicts
    preferred_batched = _get_batched(scores_preferred_metrics)
    unpreferred_batched = _get_batched(scores_unpreferred_metrics)
    
    return {'pref': preferred_batched, 'unpref': unpreferred_batched}

def data_collator_fn_for_human_feedback(batch):
    scores_preferred_metrics, scores_unpreferred_metrics = tuple(zip(*batch))
    # scores_preferred_metrics, scores_unpreferred_metrics --> list/tuple of dicts
    preferred_batched = _get_batched(scores_preferred_metrics)
    unpreferred_batched = _get_batched(scores_unpreferred_metrics)
    
    return {'pref': preferred_batched, 'unpref': unpreferred_batched}
    