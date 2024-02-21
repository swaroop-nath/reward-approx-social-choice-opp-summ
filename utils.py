from rouge import Rouge
from nltk import PorterStemmer

stemmer = PorterStemmer()
def get_rouge_score(pred_summaries, true_summaries):
    valid_ids = []
    total = len(pred_summaries)
    for idx, pred_summary in enumerate(pred_summaries):
        if len(pred_summary) > 0: valid_ids.append(idx)
    if len(valid_ids) == 0:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-L': 0.0}
    pred_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for idx, line in enumerate(pred_summaries) if idx in valid_ids]
    true_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for idx, line in enumerate(true_summaries) if idx in valid_ids]
    scorer = Rouge()
    scores = scorer.get_scores(pred_summaries, true_summaries, avg=True)

    return {'rouge-1': (scores['rouge-1']['f'] * len(valid_ids) + 0) / total, 'rouge-2': (scores['rouge-2']['f'] * len(valid_ids) + 0) / total, 'rouge-L': (scores['rouge-l']['f'] * len(valid_ids) + 0) / total}