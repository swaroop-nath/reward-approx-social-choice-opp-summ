from torch.utils.data import Dataset, DataLoader
import json

class ReviewsDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as file:
            dataset = file.readlines()
            
        self.dataset = self._process_dataset(dataset) # A list of JSON objects
        self.size = len(self.dataset)
        
    def _process_dataset(self, dataset):
        return [json.loads(line) for line in dataset]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        reviews = item['reviews']
        review_text = ' '.join(reviews)
        summary = item['summary']
        score = item.get('score', 0.0) # 0 score when no scores are available -- 0 RL loss, boils down to a slow SL.
        
        return review_text, summary, score
    
def _tokenize_text(tokenizer, text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True) # Truncate to max length supported by the model
    
def _data_collator_fn(batch, tokenizer):
    # batch is a list of 3-tuple
    review_texts, summaries, scores = tuple(zip(*batch))
    review_texts = list(review_texts)
    summaries = list(summaries)
    scores = list(scores)
    
    review_texts_tokenized = _tokenize_text(tokenizer, review_texts)
    summaries_tokenized = _tokenize_text(tokenizer, summaries)
    
    return {
        'reviews-input-ids': review_texts_tokenized['input_ids'], # tensor
        'reviews-attention-mask': review_texts_tokenized['attention_mask'], # tensor
        'sumamries-input-ids': summaries_tokenized['input_ids'][:, :-1], # tensor
        'summaries-attention-mask': summaries_tokenized['attention_mask'][:, :-1], # tensor
        'gt-summaries': summaries_tokenized['input_ids'][:, 1:], # tensor
        'summaries-scores': scores # list
    }
    
def get_data_loader(data_path, tokenizer, batch_size, is_train=True):
    dataset = ReviewsDataset(data_path)
    data_loader = DataLoader(dataset, batch_size, shuffle=is_train, collate_fn=lambda batch: _data_collator_fn(batch, tokenizer))
    
    return data_loader