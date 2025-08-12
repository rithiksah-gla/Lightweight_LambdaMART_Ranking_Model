# In-Context Learning (ICL) + Learning-to-Rank (LeToR) Pipeline

## 1. Overview
This repository implements multiple **In-Context Learning (ICL)** pipelines combined with **Learning-to-Rank (LeToR)** using **LambdaMART** to improve demonstration selection for Large Language Models (LLMs).  
We experiment with different feature sets (word, topics, interactions, phi) and embedding sources (LDA, Word2Vec) to compare performance across multiple example ordering strategies.

**Core idea**:
1. Prepare LLM utilities.
2. Retrieve **top-50** candidate examples for each test query using SBERT.
3. Extract features (basic, topic, interaction, phi) from query–candidate pairs.
4. Train a LambdaMART ranker.
5. Use the trained ranker to select **top-k** demonstrations for ICL inference with LLaMA.
6. Evaluate using Accuracy, Precision, Recall, and F1-score.


## General Pipeline Steps:

1. **Data Preparation**  
Run these file for LLM utilities:
   python Data_Prep_Train.py
   python Data_Prep_Test.py
   python Data_Prep_Val.py

2. **Top-50 Retrieval + Feature Extraction**
Choose the appropriate Test_Instance_Top50_*.py for your feature set.

4. 
