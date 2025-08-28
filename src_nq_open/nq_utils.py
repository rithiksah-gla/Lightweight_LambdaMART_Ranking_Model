import torch
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from nltk.util import ngrams
import nltk
from typing import List, Dict
import re
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from collections import Counter
import pyterrier as pt
nltk.download('punkt')
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm

# Load LLM model and tokenizer globally
hf_token = "hf***"  # Replace with your actual token
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sanitize_query(query: str) -> str:
    """Sanitize query by replacing special characters with spaces."""
    query = query.lower()
    query = re.sub(r'["\'(){}<>\\|/:*?&#=-]', ' ', query)  # Your previous sanitization
    query = re.sub(r"\s+", " ", query).strip()  # Collapse multiple spaces
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
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
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
    return pairwise_prf1(pred_answer, gold_answers)  # Use token-based F1 instead of trigrams


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
    # Build IDF over stemmed text
    stemmed_texts = [" ".join(tokenize(t)) for t in text_list]
    vec = TfidfVectorizer()
    vec.fit(stemmed_texts)
    vocab = vec.get_feature_names_out()
    idfs  = vec.idf_
    return dict(zip(vocab, idfs))


# NEW: build IDF from a Terrier index by streaming meta["text"].
def build_idf_from_terrier_index(index_dir: str,
                                 meta_field: str = "text",
                                 sample_docs: int = None) -> dict:
    """
    Stream docs from a Terrier index and compute IDF:
      idf(t) = log((N + 1) / (df(t) + 1)) + 1
    - Works in constant memory (only keeps a Counter + token set per doc).
    - If sample_docs is set, stops after that many docs (for a quick IDF).
    """
    if not pt.started():
        pt.init(logging="WARN")

    index = pt.IndexFactory.of(index_dir)
    meta = index.getMetaIndex()
    N = int(index.getCollectionStatistics().getNumberOfDocuments())

    if sample_docs is not None:
        N = min(N, int(sample_docs))

    df = Counter()
    # stream each doc's text, tokenize, update unique-term DF
    for docid in range(N):
        try:
            raw = meta.getItem(meta_field, docid)
        except Exception:
            raw = ""
        terms = set(tokenize(raw))  # unique tokens per doc
        df.update(terms)

    # compute IDF
    idf = {t: math.log((N + 1.0) / (df[t] + 1.0)) + 1.0 for t in df}
    return idf

def build_w2v_model(texts, vector_size=300, window=5, min_count=2, sg=1, workers=4, epochs=5):
    """
    Train a Word2Vec model on raw texts using your existing tokenize() in nq_utils.
    Returns the trained model.
    """
    tokenized = [tokenize(t) for t in texts]
    return Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,           # 1 = skip-gram, 0 = CBOW
        workers=workers,
        epochs=epochs
    )

def extract_drmm_w2v_features(query, doc, w2v_model, num_bins=20):
    """
    DRMM-style 20-d histogram over cosine similarities between each query token vector
    and all doc token vectors (Word2Vec). Per-q-token histogram in [-1,1] with density=True,
    then average across q-tokens.
    """
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)

    # collect L2-normalised vectors only for tokens present in vocab
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

    q_mat = np.stack(q_vecs)          # [Q, dim]
    d_mat = np.stack(d_vecs)          # [D, dim]
    sim = q_mat @ d_mat.T             # cosine matrix since rows are normalised

    hists = []
    for i in range(sim.shape[0]):
        hist, _ = np.histogram(sim[i], bins=num_bins, range=(-1.0, 1.0), density=True)
        hists.append(hist)
    avg_hist = np.mean(hists, axis=0)
    return avg_hist.tolist()

