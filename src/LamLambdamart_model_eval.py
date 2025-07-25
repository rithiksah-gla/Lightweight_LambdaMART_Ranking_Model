import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

stemmer = PorterStemmer()
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens]

# Load model and test data
model = lgb.Booster(model_file="lambdamart_model.txt")
test_df = pd.read_csv("test_dataset_lambdamart.csv")

# IDF dictionary
all_texts = test_df["query"].tolist() + test_df["candidate"].tolist()
all_texts_stemmed = [" ".join(tokenize(text)) for text in all_texts]
vectorizer = TfidfVectorizer()
vectorizer.fit(all_texts_stemmed)
idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def extract_features(query, doc, idf_dict):
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    num_q_terms = len(q_tokens)
    num_q_unique = len(set(q_tokens))
    num_d_terms = len(d_tokens)
    num_d_unique = len(set(d_tokens))
    q_idfs = [idf_dict.get(t, 0) for t in q_tokens]
    d_idfs = [idf_dict.get(t, 0) for t in d_tokens]
    min_q_idf = min(q_idfs) if q_idfs else 0
    max_q_idf = max(q_idfs) if q_idfs else 0
    sum_q_idf = sum(q_idfs)
    min_d_idf = min(d_idfs) if d_idfs else 0
    max_d_idf = max(d_idfs) if d_idfs else 0
    sum_d_idf = sum(d_idfs)
    overlap = len(set(q_tokens) & set(d_tokens))
    bm25_score = sum([idf_dict.get(t, 0) for t in set(q_tokens) & set(d_tokens)])
    return [
        num_q_terms, num_q_unique, num_d_terms, num_d_unique,
        min_q_idf, max_q_idf, sum_q_idf,
        min_d_idf, max_d_idf, sum_d_idf,
        overlap, bm25_score
    ]

features = test_df.apply(lambda row: extract_features(row['query'], row['candidate'], idf_dict), axis=1, result_type='expand')
X_test = features

# Predict
test_df['pred_score'] = model.predict(X_test)
test_df['rank'] = test_df.groupby('query')['pred_score'].rank(method='first', ascending=False)

# Metrics
def compute_f1_at_k(df, k=10):
    y_true, y_pred = [], []
    for _, group in df.groupby('query'):
        top_k = group.sort_values('pred_score', ascending=False).head(k)
        preds = [1] * len(top_k)
        labels = top_k['candidate_label'].tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
    return f1_score(y_true, y_pred)

def compute_ndcg_at_k(df, k=10):
    def dcg(relevances):
        return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(relevances))
    ndcg_scores = []
    for _, group in df.groupby('query'):
        sorted_preds = group.sort_values('pred_score', ascending=False).head(k)
        sorted_truth = group.sort_values('candidate_label', ascending=False).head(k)
        dcg_val = dcg(sorted_preds['candidate_label'].tolist())
        idcg_val = dcg(sorted_truth['candidate_label'].tolist())
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def compute_precision_recall_accuracy_at_k(df, k=10):
    y_true_all, y_pred_all = [], []
    for _, group in df.groupby('query'):
        sorted_group = group.sort_values('pred_score', ascending=False)
        top_k = sorted_group.head(k)
        rest = sorted_group.tail(len(group) - k)
        preds = [1] * len(top_k) + [0] * len(rest)
        labels = top_k['candidate_label'].tolist() + rest['candidate_label'].tolist()
        y_true_all.extend(labels)
        y_pred_all.extend(preds)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    return accuracy, precision, recall

# Final scores
f1 = compute_f1_at_k(test_df, k=10)
ndcg = compute_ndcg_at_k(test_df, k=10)
accuracy, precision, recall = compute_precision_recall_accuracy_at_k(test_df, k=10)

print(f"F1@10: {f1:.4f}")
print(f"NDCG@10: {ndcg:.4f}")
print(f"Accuracy@10: {accuracy:.4f}")
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")
