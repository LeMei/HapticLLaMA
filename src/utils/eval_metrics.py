import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import Counter
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice

nltk.download('wordnet')


def compute_avg_bleu(reference_list, candidate):
    references = [ref.split() for ref in reference_list]
    candidate_tokens = candidate.split()
    
    bleu_1_scores = [
        sentence_bleu([ref], candidate_tokens, weights=(1, 0, 0, 0), 
                      smoothing_function=SmoothingFunction().method1) 
        for ref in references
    ]
    
    bleu_4_scores = [
        sentence_bleu([ref], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), 
                      smoothing_function=SmoothingFunction().method1) 
        for ref in references
    ]
    
    return np.mean(bleu_1_scores), np.std(bleu_1_scores), np.mean(bleu_4_scores), np.std(bleu_4_scores)

def compute_bleu(reference_list, candidate):
    references = [ref.split() for ref in reference_list]
    candidate_tokens = candidate.split()
    
    bleu_1_scores = [
        sentence_bleu([ref], candidate_tokens, weights=(1, 0, 0, 0), 
                      smoothing_function=SmoothingFunction().method1) 
        for ref in references
    ]
    
    bleu_4_scores = [
        sentence_bleu([ref], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), 
                      smoothing_function=SmoothingFunction().method1) 
        for ref in references
    ]
    
    return max(bleu_1_scores), min(bleu_1_scores), max(bleu_4_scores), min(bleu_4_scores)

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(candidate, ref)['rougeL'].fmeasure for ref in reference]
    avg_rouge_l = np.mean(rouge_l_scores)
    std_rouge_l = np.std(rouge_l_scores)
    return avg_rouge_l, std_rouge_l

def compute_meteor(reference_list, candidate):
    reference = [ref.split() for ref in reference_list]
    candidate_tokens = candidate.split()
    meteor = round(meteor_score(reference, candidate_tokens),4)
    return meteor

def compute_mse(y_true, y_pred):
    y_true = [t.cpu() for t in y_true]
    y_true = np.concatenate([t for t in y_true], axis=0)

    y_pred = [t.cpu() for t in y_pred]
    y_pred = np.concatenate([t for t in y_pred], axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2



# 计算 CIDEr
# def compute_cider(reference, candidate):
#     cider_scorer = Cider()
#     score, _ = cider_scorer.compute_score({0: reference}, {0: [candidate]})
#     return score[0]

# def compute_spice(reference, candidate):
#     """计算 SPICE 分数"""
#     spice_scorer = Spice()
#     score, _ = spice_scorer.compute_score({0: reference}, {0: [candidate]})
#     return score[0]

# 示例测试
# reference_texts = ["The quick brown fox jumps over the lazy dog."]
# generated_text = "A fast brown fox leaps over a sleeping dog."

# bleu1, bleu4 = compute_bleu(reference_texts, generated_text)
# rouge_l = compute_rouge(reference_texts, generated_text)
# meteor = compute_meteor(reference_texts, generated_text)
# cider = compute_cider(reference_texts, generated_text)
# spice = compute_spice(reference_texts, generated_text)

# print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")
# print(f"ROUGE-L: {rouge_l:.4f}")
# print(f"METEOR: {meteor:.4f}")
# print(f"CIDEr: {cider:.4f}")
# print(f"SPICE: {spice:.4f}")

    



