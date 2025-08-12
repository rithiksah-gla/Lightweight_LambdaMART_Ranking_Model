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
0. Run the **utils.py** file to define the repetitve methods.

1. **Data Preparation**  
Run these file for LLM utilities:
- python Data_Prep_Train.py
- python Data_Prep_Test.py
- python Data_Prep_Val.py

2. **Top-50 Retrieval + Feature Extraction**
Choose the appropriate Test_Instance_Top50_*.py for your feature set
- Test_Instance_Top50_Word.py
- Test_Instance_Top50_Word_DRMM(w2v).py
- Test_Instance_Top50_Word_DRMM.py
- Test_Instance_Top50_Word_DRMM_phi(w2v).py
- Test_Instance_Top50_Word_LDA.py
- Test_Instance_Top50_Word_LDA_DRMM.py

3. **Model Training**
Train the LambdaMART ranker for your chosen features:
- Lambdamart_Model_Training_Word.py
- Lambdamart_Model_Training_Word_DRMM(w2v).py
- Lambdamart_Model_Training_Word_DRMM.py
- Lambdamart_Model_Training_Word_DRMM_phi(w2v).py
- Lambdamart_Model_Training_Word_LDA.py
- Lambdamart_Model_Training_Word_LDA_DRMM.py

4. **ICL Inference**
- ICL_Inference_Kshot_Letor_SST2_Word.py
- ICL_Inference_Kshot_Letor_SST2_Word_DRMM(w2v).py
- ICL_Inference_Kshot_Letor_SST2_Word_DRMM.py
- ICL_Inference_Kshot_Letor_SST2_Word_DRMM_phi(w2v).py
- ICL_Inference_Kshot_Letor_SST2_Word_LDA.py
- ICL_Inference_Kshot_Letor_SST2_Word_LDA_DRMM.py
- ICL_Inference_Kshot_SST2.py (**Baseline: Unsupervised Inference**)
- ICL_Inference_Zero_Shot_SST2.py (**Baseline: Zero-Shot Inference**)
