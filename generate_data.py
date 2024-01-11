import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from numpy.random import choice

DEVICE = 'cuda:1'

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = json.load(file)
        
    return prompts

def read_reviews(file_path):
    reviews = []
    
    df = pd.read_csv(file_path)
    for r in tqdm(df.review_text, desc='Loading Data'):
        reviews.append(eval(r))
        
    return reviews

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    
    return tokenizer, model

def summarize_using_prompt(tokenizer, model, prompt_template, reviews, top_p=0.95, top_k=25, max_new_tokens=500):
    review_text = ""
    for idx, review in enumerate(reviews):
        review_text += f"{idx+1}. {review}\n"
    prompt = prompt_template.format(review_text)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p, top_k=top_k)
    return tokenizer.decode(outputs[0][inputs['input_ids'].size(1):])

def generate_summaries(reviews, model, tokenizer, prompts, num_samples, store_dir):
    index_list = list(range(len(reviews)))
    chosen_indices = choice(index_list, num_samples, replace=False)
    
    for index in tqdm(chosen_indices, desc='Generating Summaries'):
        prod_reviews = reviews[index]
        model_output_good_1 = summarize_using_prompt(tokenizer, model, prompts['GOOD'], prod_reviews, top_p=0.85, top_k=2, max_new_tokens=200)
        model_output_good_2 = summarize_using_prompt(tokenizer, model, prompts['GOOD'], prod_reviews, top_p=0.85, top_k=2, max_new_tokens=200)
        model_output_good_3 = summarize_using_prompt(tokenizer, model, prompts['GOOD'], prod_reviews, top_p=0.85, top_k=2, max_new_tokens=200)
        model_output_sbad_1 = summarize_using_prompt(tokenizer, model, prompts['SLIGHTLY-BAD'], prod_reviews, top_p=0.90, top_k=50, max_new_tokens=200)
        model_output_sbad_2 = summarize_using_prompt(tokenizer, model, prompts['SLIGHTLY-BAD'], prod_reviews, top_p=0.90, top_k=50, max_new_tokens=200)
        model_output_sbad_3 = summarize_using_prompt(tokenizer, model, prompts['SLIGHTLY-BAD'], prod_reviews, top_p=0.90, top_k=50, max_new_tokens=200)
        model_output_vbad_1 = summarize_using_prompt(tokenizer, model, prompts['VERY-BAD'], prod_reviews, top_p=0.99, top_k=1000, max_new_tokens=200)
        model_output_vbad_2 = summarize_using_prompt(tokenizer, model, prompts['VERY-BAD'], prod_reviews, top_p=0.99, top_k=1000, max_new_tokens=200)
        model_output_vbad_3 = summarize_using_prompt(tokenizer, model, prompts['VERY-BAD'], prod_reviews, top_p=0.99, top_k=1000, max_new_tokens=200)
        
        storage_loc = store_dir + f'/index-{index}'
        if not os.path.exists(storage_loc): os.makedirs(storage_loc)
        
        with open(f'{storage_loc}/good-1.txt', 'w') as file:
            file.write(model_output_good_1)
        
        with open(f'{storage_loc}/good-2.txt', 'w') as file:
            file.write(model_output_good_2)
            
        with open(f'{storage_loc}/good-3.txt', 'w') as file:
            file.write(model_output_good_3)
        
        with open(f'{storage_loc}/sbad-1.txt', 'w') as file:
            file.write(model_output_sbad_1)
            
        with open(f'{storage_loc}/sbad-2.txt', 'w') as file:
            file.write(model_output_sbad_2)
            
        with open(f'{storage_loc}/sbad-3.txt', 'w') as file:
            file.write(model_output_sbad_3)
        
        with open(f'{storage_loc}/vbad-1.txt', 'w') as file:
            file.write(model_output_vbad_1)
            
        with open(f'{storage_loc}/vbad-2.txt', 'w') as file:
            file.write(model_output_vbad_2)
        
        with open(f'{storage_loc}/vbad-3.txt', 'w') as file:
            file.write(model_output_vbad_3)

if __name__ == '__main__':
    prompts = load_prompts('./prompts.json')
    train_reviews = read_reviews('./amazon_train_data.csv')
    val_reviews = read_reviews('./amazon_val_data.csv')
    tokenizer, model = load_model('./models/Mistral-7B-Instruct-v0.2/model_files')
    
    TRAIN_SAMPLE = 12500
    VAL_SAMPLE = 5000
    
    TRAIN_STORE_PATH = './instruction-data-v2/training'
    VAL_STORE_PATH = './instruction-data-v2/validation'
    
    if not os.path.exists(TRAIN_STORE_PATH): os.makedirs(TRAIN_STORE_PATH)
    if not os.path.exists(VAL_STORE_PATH): os.makedirs(VAL_STORE_PATH)

    generate_summaries(train_reviews, model, tokenizer, prompts, TRAIN_SAMPLE, TRAIN_STORE_PATH)
    generate_summaries(val_reviews, model, tokenizer, prompts, VAL_SAMPLE, VAL_STORE_PATH)