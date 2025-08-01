# utils.py
import re
import torch
from torch.nn import functional as F
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokenizer and Stemmer
stemmer = PorterStemmer()

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens]

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

    overlap = set(q_tokens) & set(d_tokens)
    bm25_score = sum([idf_dict.get(t, 0) for t in overlap])

    return [
	num_q_terms, num_q_unique, num_d_terms, num_d_unique,
        min_q_idf, max_q_idf, sum_q_idf,
        min_d_idf, max_d_idf, sum_d_idf,
        len(overlap), bm25_score
    ]

def get_idf_dict(text_list):
    stemmed_texts = [" ".join(tokenize(text)) for text in text_list]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(stemmed_texts)
    feature_names = (
    vectorizer.get_feature_names_out()
    if hasattr(vectorizer, "get_feature_names_out")
    else vectorizer.get_feature_names()
    )
    return dict(zip(feature_names, vectorizer.idf_))

def score_candidate_llm(query_text, query_label, candidate_text, candidate_label, tokenizer, model, device, prompt_prefix):
    # Build ICL prompt
    demonstration = f"Review: {candidate_text}\nSentiment: {'positive' if candidate_label else 'negative'}\n"
    prompt = f"{prompt_prefix}{demonstration}Review: {query_text}\nSentiment: "

    #Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Run through model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Focus on last token (sentiment prediction)
    last_token_logits = logits[:, -1, :]
    class_tokens = [
        tokenizer.encode("negative", add_special_tokens=False)[0],
        tokenizer.encode("positive", add_special_tokens=False)[0]
    ]

    probs = F.softmax(last_token_logits[:, class_tokens], dim=-1)
    # Get probability of the correct label
    return probs[0, query_label].item()

def load_sst2_data(path):
    examples = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            examples.append({"text": text, "label": int(label)})
    return examples
