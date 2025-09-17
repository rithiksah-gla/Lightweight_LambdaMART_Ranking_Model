# In-Context Learning (ICL) + Learning-to-Rank (LeToR) Pipeline

## 1. Overview

This repository implements multiple **In-Context Learning (ICL)** and **Retrieval-Augmented Generation (RAG)** pipelines combined with **Learning-to-Rank (LeToR)** using **LambdaMART**.

We experiment with different feature sets (word, topics, interactions, phi, DRMM histograms) and embedding sources (LDA, Word2Vec) to compare performance across multiple example ordering strategies.

**Core idea**:

1. Prepare LLM utilities as supervision signals.
2. Retrieve **top-50** candidate examples/passages for each test query using SBERT.
3. Extract features (lexical, topic, interaction, phi, DRMM) from query–candidate pairs.
4. Train a LambdaMART ranker.
5. Use the trained ranker to select **top-k** contexts.
6. Run inference with LLaMA-2 on SST-2 (ICL) and NQ-Open (RAG).
7. Evaluate using Accuracy/Precision/Recall/F1 (SST-2) and F1 (NQ-Open).

---

## 2. General Pipeline Steps

0. Run the **utils.py** file to define the shared feature extraction methods.

1. **Data Preparation**
   Generate supervision signals from LLM utilities:

```bash
python Data_Prep_Train.py
python Data_Prep_Test.py
python Data_Prep_Val.py
```

2. **Top-50 Retrieval + Feature Extraction**
   Choose the appropriate file for your feature set:

* `Test_Instance_Top50_Word.py`
* `Test_Instance_Top50_Word_DRMM(w2v).py`
* `Test_Instance_Top50_Word_DRMM.py`
* `Test_Instance_Top50_Word_DRMM_phi(w2v).py`
* `Test_Instance_Top50_Word_LDA.py`
* `Test_Instance_Top50_Word_LDA_DRMM.py`

3. **Model Training**
   Train the LambdaMART ranker with chosen features:

* `Lambdamart_Model_Training_Word.py`
* `Lambdamart_Model_Training_Word_DRMM(w2v).py`
* `Lambdamart_Model_Training_Word_DRMM.py`
* `Lambdamart_Model_Training_Word_DRMM_phi(w2v).py`
* `Lambdamart_Model_Training_Word_LDA.py`
* `Lambdamart_Model_Training_Word_LDA_DRMM.py`

4. **ICL Inference (SST-2)**

* `ICL_Inference_Kshot_Letor_SST2_Word.py`
* `ICL_Inference_Kshot_Letor_SST2_Word_DRMM(w2v).py`
* `ICL_Inference_Kshot_Letor_SST2_Word_DRMM.py`
* `ICL_Inference_Kshot_Letor_SST2_Word_DRMM_phi(w2v).py`
* `ICL_Inference_Kshot_Letor_SST2_Word_LDA.py`
* `ICL_Inference_Kshot_Letor_SST2_Word_LDA_DRMM.py`
* `ICL_Inference_Kshot_SST2.py` (**Baseline: Unsupervised**)
* `ICL_Inference_Zero_Shot_SST2.py` (**Baseline: Zero-Shot**)

---

## 3. NQ-Open (RAG) Pipeline

We extend the same ranking framework to **NQ-Open**, a large-scale open-domain QA benchmark.
👉 Repo: [src\_nq\_open](https://github.com/rithiksah-gla/Lightweight_LambdaMART_Ranking_Model/tree/main/src_nq_open)

### Steps:

1. **Data Preparation**
   Generate training/test/validation supervision labels with:

```bash
python Data_Prep_Train_NQ.py
python Data_Prep_Test_NQ.py
python Data_Prep_Val_NQ.py
```

2. **Top-50 Retrieval + Feature Extraction**
   Candidate passages retrieved using SBERT, features extracted with:

* `Test_Instance_Top50_NQ_Word.py`
* `Test_Instance_Top50_NQ_Word_DRMM.py`
* `Test_Instance_Top50_NQ_Word_DRMM(w2v).py`
* `Test_Instance_Top50_NQ_Word_LDA.py`
* `Test_Instance_Top50_NQ_Word_LDA_DRMM.py`

3. **Model Training**
   Train LambdaMART on extracted features:

* `Lambdamart_Model_Training_NQ_Word.py`
* `Lambdamart_Model_Training_NQ_Word_DRMM.py`
* `Lambdamart_Model_Training_NQ_Word_DRMM(w2v).py`
* `Lambdamart_Model_Training_NQ_Word_LDA.py`
* `Lambdamart_Model_Training_NQ_Word_LDA_DRMM.py`

4. **RAG Inference**
   Run evaluation with trained rankers:

* `RAG_Inference_Kshot_Letor_NQ_Word.py`
* `RAG_Inference_Kshot_Letor_NQ_Word_DRMM.py`
* `RAG_Inference_Kshot_Letor_NQ_Word_DRMM(w2v).py`
* `RAG_Inference_Kshot_Letor_NQ_Word_LDA.py`
* `RAG_Inference_Kshot_Letor_NQ_Word_LDA_DRMM.py`
* `RAG_Inference_Kshot_NQ.py` (**Baseline: Unsupervised**)
* `RAG_Inference_Zero_Shot_NQ.py` (**Baseline: Zero-Shot**)

### Metrics:

* Evaluated with **F1-score** (token overlap with gold answers).
* Feature-rich LambdaMART ranker shows consistent improvements over baselines.

---

## 4. Key Features

* Shared **feature extraction** code (`utils.py`) across ICL and RAG.
* Supports **lexical (IDF, BM25)**, **topic (LDA)**, **interaction (DRMM histograms)**, and **norm-based** features.
* Flexible training/inference scripts for both **SST-2 (ICL)** and **NQ-Open (RAG)**.
