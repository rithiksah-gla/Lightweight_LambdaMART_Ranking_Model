import torch
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from nltk.util import ngrams
import nltk
from typing import List, Dict
import re, string, math
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pyterrier as pt
nltk.download('punkt')
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from gensim import corpora, models
from gensim.matutils import sparse2full
from itertools import product

# ------------------------------
# LLM (unchanged)
# ------------------------------
hf_token = "hf***"  # (update with your token)
model_name = 'meta-llama/llama-2-7b-hf'
single_precision = True

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if single_precision else torch.float32,
    use_auth_token=hf_token,
    cache_dir="/scratch/2980356s/hf_cache/"
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    use_auth_token=hf_token,
    cache_dir="/scratch/2980356s/hf_cache/"
)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ------------------------------
# Normalisation / utilities (unchanged)
# ------------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sanitize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r'["\'(){}<>\\|/:*?&#=-]', ' ', query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

def contains_any_answer(text: str, golds: List[str]) -> bool:
    norm_text = normalize_text(text)
    return any(normalize_text(g) in norm_text for g in golds)

def pairwise_prf1(pred_text: str, gold_answers: List[str]) -> Dict[str, float]:
    pred_tokens = normalize_text(pred_text).split()
    gold_tokens = normalize_text(" ".join(gold_answers)).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common_vocab = set(pred_tokens) & set(gold_tokens)
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common_vocab)
    precision = num_common / len(pred_tokens) if len(pred_tokens) else 0.0
    recall = num_common / len(gold_tokens) if len(gold_tokens) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

def predict_answer(query: str, text: str) -> str:
    task_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-$"
    documents_str = f"Document: {text.lower()}\n"
    prompt = f"{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query.lower()}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=1, early_stopping=True)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = predicted_answer.find("Answer:") + len("Answer:")
    return predicted_answer[answer_start:].strip() if answer_start != -1 else predicted_answer.strip()

def compute_f1(pred_answer: str, gold_answers: List[str]) -> Dict[str, float]:
    return pairwise_prf1(pred_answer, gold_answers)

# ------------------------------
# Tokeniser + 12 word features (unchanged)
# ------------------------------
stemmer = PorterStemmer()
def tokenize(text: str):
    tokens = re.findall(r'\b\w+\b', (text or "").lower())
    return [stemmer.stem(t) for t in tokens]

def extract_features(query: str, doc: str, idf_dict: dict):
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    num_q_terms   = len(q_tokens)
    num_q_unique  = len(set(q_tokens))
    num_d_terms   = len(d_tokens)
    num_d_unique  = len(set(d_tokens))
    q_idfs = [idf_dict.get(t, 0.0) for t in q_tokens]
    d_idfs = [idf_dict.get(t, 0.0) for t in d_tokens]
    min_q_idf = min(q_idfs) if q_idfs else 0.0
    max_q_idf = max(q_idfs) if q_idfs else 0.0
    sum_q_idf = sum(q_idfs)
    min_d_idf = min(d_idfs) if d_idfs else 0.0
    max_d_idf = max(d_idfs) if d_idfs else 0.0
    sum_d_idf = sum(d_idfs)
    overlap = set(q_tokens) & set(d_tokens)
    bm25_score_proxy = sum(idf_dict.get(t, 0.0) for t in overlap)
    return [
        num_q_terms, num_q_unique, num_d_terms, num_d_unique,
        min_q_idf, max_q_idf, sum_q_idf,
        min_d_idf, max_d_idf, sum_d_idf,
        len(overlap), bm25_score_proxy
    ]

def get_idf_dict(text_list):
    stemmed_texts = [" ".join(tokenize(t)) for t in text_list]
    vec = TfidfVectorizer()
    vec.fit(stemmed_texts)
    vocab = vec.get_feature_names_out()
    idfs  = vec.idf_
    return dict(zip(vocab, idfs))

def build_idf_from_terrier_index(index_dir: str,
                                 meta_field: str = "text",
                                 sample_docs: int = None) -> dict:
    if not pt.started():
        pt.init(logging="WARN")
    index = pt.IndexFactory.of(index_dir)
    meta = index.getMetaIndex()
    N = int(index.getCollectionStatistics().getNumberOfDocuments())
    if sample_docs is not None:
        N = min(N, int(sample_docs))
    df = Counter()
    for docid in range(N):
        try:
            raw = meta.getItem(meta_field, docid)
        except Exception:
            raw = ""
        terms = set(tokenize(raw))
        df.update(terms)
    idf = {t: math.log((N + 1.0) / (df[t] + 1.0)) + 1.0 for t in df}
    return idf

# ------------------------------
# W2V: build + DRMM features (unchanged)
# ------------------------------
def build_w2v_model(texts, vector_size=300, window=5, min_count=2, sg=1, workers=4, epochs=5):
    tokenized = [tokenize(t) for t in texts]
    return Word2Vec(
        sentences=tokenized, vector_size=vector_size, window=window,
        min_count=min_count, sg=sg, workers=workers, epochs=epochs
    )

def extract_drmm_w2v_features(query, doc, w2v_model, num_bins=20):
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    q_vecs, d_vecs = [], []
    for t in q_tokens:
        if t in w2v_model.wv:
            v = w2v_model.wv[t]; n = norm(v)
            if n > 0: q_vecs.append(v / n)
    for t in d_tokens:
        if t in w2v_model.wv:
            v = w2v_model.wv[t]; n = norm(v)
            if n > 0: d_vecs.append(v / n)
    if not q_vecs or not d_vecs:
        return [0.0] * num_bins
    q_mat = np.stack(q_vecs)
    d_mat = np.stack(d_vecs)
    sim = q_mat @ d_mat.T
    hists = []
    for i in range(sim.shape[0]):
        hist, _ = np.histogram(sim[i], bins=num_bins, range=(-1.0, 1.0), density=True)
        hists.append(hist)
    avg_hist = np.mean(hists, axis=0)
    return avg_hist.tolist()

# ------------------------------
# LDA: build + 16 topic features  (NEW for Word+LDA)
# ------------------------------
def build_lda_model(texts, num_topics=50):
    tokenized_texts = [tokenize(t) for t in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, dictionary

def extract_lda_features(query, doc, lda_model, lda_dict, num_topics=50):
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
            for i, a in enumerate(vecs)
            for j, b in enumerate(vecs) if i < j and norm(a) > 0 and norm(b) > 0
        ]
        if not sims:
            return 0.0, 0.0, 0.0
        return min(sims), max(sims), float(np.mean(sims))

    def cross_sim_stats(q_vecs, d_vecs):
        sims = [
            np.dot(q, d) / (norm(q) * norm(d))
            for q in q_vecs for d in d_vecs
            if norm(q) > 0 and norm(d) > 0
        ]
        if not sims:
            return 0.0, 0.0, 0.0
        return min(sims), max(sims), float(np.mean(sims))

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0.0

    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)

    q_phi = get_phi_vectors(q_tokens)
    d_phi = get_phi_vectors(d_tokens)

    q_theta = get_theta_vector(q_tokens)
    d_theta = get_theta_vector(d_tokens)

    q_avg_phi = avg_phi(q_phi)
    q_min_norm, q_max_norm = norm_stats(q_phi)
    q_min_sim, q_max_sim, q_avg_sim = pairwise_sim_stats(q_phi)

    d_avg_phi = avg_phi(d_phi)
    d_min_norm, d_max_norm = norm_stats(d_phi)
    d_min_sim, d_max_sim, d_avg_sim = pairwise_sim_stats(d_phi)

    sl_sim, cl_sim, al_sim = cross_sim_stats(q_phi, d_phi)
    theta_sim = cosine_sim(q_theta, d_theta)

    return [
        float(norm(q_avg_phi)),  # 1
        q_min_norm,              # 2
        q_max_norm,              # 3
        q_min_sim,               # 4
        q_max_sim,               # 5
        q_avg_sim,               # 6
        float(norm(d_avg_phi)),  # 7
        d_min_norm,              # 8
        d_max_norm,              # 9
        d_min_sim,               # 10
        d_max_sim,               # 11
        d_avg_sim,               # 12
        sl_sim,                  # 13
        cl_sim,                  # 14
        al_sim,                  # 15
        theta_sim                # 16
    ]

def extract_drmm_features(query, doc, lda_model, lda_dict, num_topics=50, num_bins=20):
    """
    DRMM-style 20-bin histogram interaction using LDA token-level topic vectors (phi).
    For each query token, compute cosine sims to ALL doc-token topic vectors,
    histogram sims into [-1,1] with density=True, then average histograms over query tokens.
    Returns a 20-d list.
    """
    import numpy as np

    def token_phi_vecs(tokens):
        if not tokens:
            return []
        bow = lda_dict.doc2bow(tokens)
        vecs = []
        # build a phi vector per token present in this text
        for token_id, count in bow:
            single = [(token_id, count)]
            topic_dist = lda_model.get_document_topics(single, minimum_probability=0)
            vec = sparse2full(topic_dist, num_topics)  # length=num_topics
            vnorm = norm(vec)
            if vnorm > 0:
                vec = vec / vnorm
            vecs.append(vec)
        return vecs

    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    q_phi = token_phi_vecs(q_tokens)
    d_phi = token_phi_vecs(d_tokens)

    if not q_phi or not d_phi:
        return [0.0] * num_bins

    q_mat = np.stack(q_phi)   # [Q, num_topics]
    d_mat = np.stack(d_phi)   # [D, num_topics]
    # cosine since rows are L2-normalised:
    sim = q_mat @ d_mat.T

    hists = []
    for i in range(sim.shape[0]):
        hist, _ = np.histogram(sim[i], bins=num_bins, range=(-1.0, 1.0), density=True)
        hists.append(hist)
    avg_hist = np.mean(hists, axis=0)
    return avg_hist.tolist()


def extract_drmm_features(query, doc, lda_model, lda_dict, num_topics=50, num_bins=20):
    """
    DRMM-style 20-bin histogram features using LDA token-level phi vectors.
    - For each token in query/doc (via dictionary BOW), get a topic (phi) vector
      by calling LDA on a single-token BoW and expanding to dense with sparse2full.
    - Build cosine similarity matrix between all query-token phi vectors and doc-token phi vectors.
    - For each query token: histogram similarities into num_bins in [-1,1], density=True.
    - Average the per-q-token histograms -> 20-d feature vector.
    """
    def token_phi_vecs(tokens):
        bow = lda_dict.doc2bow(tokens)
        vecs = []
        for token_id, count in bow:
            single = [(token_id, count)]
            topic_dist = lda_model.get_document_topics(single, minimum_probability=0)
            phi = sparse2full(topic_dist, num_topics)
            # normalise to unit length (defensive: zero-safe)
            n = norm(phi)
            vecs.append(phi / n if n > 0 else phi)
        return vecs

    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    q_phi = token_phi_vecs(q_tokens)
    d_phi = token_phi_vecs(d_tokens)

    if not q_phi or not d_phi:
        return [0.0] * num_bins

    q_mat = np.stack(q_phi)  # [Q, T]
    d_mat = np.stack(d_phi)  # [D, T]
    # cosine since rows are L2-normalised
    sim = q_mat @ d_mat.T    # [Q, D]

    hists = []
    for i in range(sim.shape[0]):
        hist, _ = np.histogram(sim[i], bins=num_bins, range=(-1.0, 1.0), density=True)
        hists.append(hist)
    avg_hist = np.mean(hists, axis=0)
    return avg_hist.tolist()

def extract_phi_w2v_features(query, doc, w2v_model):
    """
    All vectors L2-normalized before similarity; cosine reduces to dot.
    """
    def get_normed_vecs(tokens):
        vecs = []
        for t in tokens:
            if t in w2v_model.wv:
                v = w2v_model.wv[t]
                n = norm(v)
                if n > 0:
                    vecs.append(v / n)
        return vecs

    def avg_vec(vecs, dim):
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)

    def norm_stats(vecs):
        if not vecs:
            return 0.0, 0.0
        ns = [float(norm(v)) for v in vecs]
        return (min(ns), max(ns)) if ns else (0.0, 0.0)

    def pairwise_sim_stats(vecs):
        if len(vecs) < 2:
            return 0.0, 0.0, 0.0
        sims = [float(np.dot(a, b))
                for i, a in enumerate(vecs)
                for j, b in enumerate(vecs) if i < j]
        return (min(sims), max(sims), float(np.mean(sims))) if sims else (0.0, 0.0, 0.0)

    def cross_link_stats(q_vecs, d_vecs):
        if not q_vecs or not d_vecs:
            return 0.0, 0.0, 0.0
        sims = [float(np.dot(q, d)) for q in q_vecs for d in d_vecs]
        return (min(sims), max(sims), float(np.mean(sims))) if sims else (0.0, 0.0, 0.0)

    # tokenize + collect vectors
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    q_vecs = get_normed_vecs(q_tokens)
    d_vecs = get_normed_vecs(d_tokens)

    dim = w2v_model.vector_size

    # query stats
    q_avg = avg_vec(q_vecs, dim)
    q_min_norm, q_max_norm = norm_stats(q_vecs)
    q_min_sim, q_max_sim, q_avg_sim = pairwise_sim_stats(q_vecs)

    # doc stats
    d_avg = avg_vec(d_vecs, dim)
    d_min_norm, d_max_norm = norm_stats(d_vecs)
    d_min_sim, d_max_sim, d_avg_sim = pairwise_sim_stats(d_vecs)

    # cross-link stats
    sl_sim, cl_sim, al_sim = cross_link_stats(q_vecs, d_vecs)

    return [
        float(norm(q_avg)),  # 1: ||avg q||
        q_min_norm,          # 2
        q_max_norm,          # 3
        q_min_sim,           # 4
        q_max_sim,           # 5
        q_avg_sim,           # 6
        float(norm(d_avg)),  # 7: ||avg d||
        d_min_norm,          # 8
        d_max_norm,          # 9
        d_min_sim,           #10
        d_max_sim,           #11
        d_avg_sim,           #12
        sl_sim,              #13 single-link (min cross)
        cl_sim,              #14 complete-link (max cross)
        al_sim               #15 average-link (mean cross)
    ]
