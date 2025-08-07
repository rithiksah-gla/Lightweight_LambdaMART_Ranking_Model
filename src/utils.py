#latest utils.py
import re
import torch
from torch.nn import functional as F
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.matutils import cossim
import numpy as np
from gensim.matutils import sparse2full
from itertools import product
from numpy.linalg import norm

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
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def score_candidate_llm(query_text, query_label, candidate_text, candidate_label, tokenizer, model, device, prompt_prefix):
    # Build ICL prompt
    demonstration = f"Review: {candidate_text}\nSentiment: {'positive' if candidate_label else 'negative'}\n"
    prompt = f"{prompt_prefix}{demonstration}Review: {query_text}\nSentiment:"
    
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


def build_lda_model(texts, num_topics=50):
    tokenized_texts = [tokenize(t) for t in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, dictionary

def extract_lda_features(query, doc, lda_model, lda_dict, num_topics=50):
    from gensim.matutils import sparse2full
    from numpy.linalg import norm
    from itertools import product
    import numpy as np

    def get_phi_vectors(tokens):
        if not tokens:
            return []
        bow = lda_dict.doc2bow(tokens)
        vecs = []
        for token_id, count in bow:
            single_token_bow = [(token_id, count)]
            topic_dist = lda_model.get_document_topics(single_token_bow, minimum_probability=0)
            vec = sparse2full(topic_dist, num_topics)
            vecs.append(vec)
        return vecs

    def get_theta_vector(tokens):
        bow = lda_dict.doc2bow(tokens)
        return sparse2full(lda_model[bow], num_topics)

    def avg_phi(vecs):
        return np.mean(vecs, axis=0) if vecs else np.zeros(num_topics)

    def norm_stats(vecs):
        norms = [norm(v) for v in vecs if norm(v) > 0]
        if not norms:
            return 0.0, 0.0
        return min(norms), max(norms)

    def pairwise_sim_stats(vecs):
        sims = [
            np.dot(a, b) / (norm(a) * norm(b))
            for a, b in product(vecs, repeat=2)
            if not np.array_equal(a, b) and norm(a) > 0 and norm(b) > 0
        ]
        if not sims:
            return 0.0, 0.0, 0.0
        return min(sims), max(sims), np.mean(sims)

    def cross_sim_stats(q_vecs, d_vecs):
        sims = [
            np.dot(q, d) / (norm(q) * norm(d))
            for q, d in product(q_vecs, d_vecs)
            if norm(q) > 0 and norm(d) > 0
        ]
        if not sims:
            return 0.0, 0.0, 0.0
        return min(sims), max(sims), np.mean(sims)

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0.0

    # Tokenize
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)

    # Get phi vectors
    q_phi = get_phi_vectors(q_tokens)
    d_phi = get_phi_vectors(d_tokens)

    # Get theta vectors
    q_theta = get_theta_vector(q_tokens)
    d_theta = get_theta_vector(d_tokens)

    # Query stats
    q_avg_phi = avg_phi(q_phi)
    q_min_norm, q_max_norm = norm_stats(q_phi)
    q_min_sim, q_max_sim, q_avg_sim = pairwise_sim_stats(q_phi)

    # Document stats
    d_avg_phi = avg_phi(d_phi)
    d_min_norm, d_max_norm = norm_stats(d_phi)
    d_min_sim, d_max_sim, d_avg_sim = pairwise_sim_stats(d_phi)

    # Cross stats
    sl_sim, cl_sim, al_sim = cross_sim_stats(q_phi, d_phi)

    # Theta cosine similarity
    theta_sim = cosine_sim(q_theta, d_theta)

    # Return final 16-dimensional feature vector
    return [
     norm(q_avg_phi),                 # 1 - norm of avg query phi vec
     q_min_norm,                     # 2
     q_max_norm,                     # 3
     q_min_sim,                      # 4
     q_max_sim,                      # 5
     q_avg_sim,                      # 6
     norm(d_avg_phi),                # 7 - norm of avg doc phi vec
     d_min_norm,                     # 8
     d_max_norm,                     # 9
     d_min_sim,                      # 10
     d_max_sim,                      # 11
     d_avg_sim,                      # 12
     sl_sim,                         # 13
     cl_sim,                         # 14
     al_sim,                         # 15
     theta_sim                       # 16
]

def extract_drmm_features(query, doc, lda_model, lda_dict, num_topics=50, num_bins=20):
    def get_phi_vectors(tokens):
        bow = lda_dict.doc2bow(tokens)
        phi_vecs = []
        for word_id, _ in bow:
            try:
                vec = sparse2full(lda_model[[word_id]], num_topics)[0]
                phi_vecs.append(vec)
            except:
                phi_vecs.append(np.zeros(num_topics))
        return phi_vecs

    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    q_phi = get_phi_vectors(q_tokens)
    d_phi = get_phi_vectors(d_tokens)

    if not q_phi or not d_phi:
        return [0] * num_bins

    sim_matrix = np.zeros((len(q_phi), len(d_phi)))
    for i, qv in enumerate(q_phi):
        for j, dv in enumerate(d_phi):
            if norm(qv) > 0 and norm(dv) > 0:
                sim_matrix[i, j] = np.dot(qv, dv) / (norm(qv) * norm(dv))

    histograms = []
    for row in sim_matrix:
        hist, _ = np.histogram(row, bins=num_bins, range=(-1, 1), density=True)
        histograms.append(hist)

    avg_histogram = np.mean(histograms, axis=0)
    return avg_histogram.tolist()
