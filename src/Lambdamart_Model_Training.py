#lambdamart training latest for 12 features

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from utils import tokenize, extract_features, get_idf_dict

# Load raw data (N=50 candidates per query)
train_df_full = pd.read_csv("/kaggle/input/new-utilities/train_dataset_lambdamart_v01_N50.csv")
val_df_full = pd.read_csv("/kaggle/input/new-utilities/validation_dataset_lambdamart_v01_N50.csv")

# Load raw train.tsv file to use for corpus (query + candidate)
train_corpus = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])

# Step 1: Select top-10 by score (for both train and val)
def get_top10_by_score(df):
    return df.sort_values(by=["query", "score"], ascending=[True, False]) \
             .groupby("query").head(10).reset_index(drop=True)

train_top10 = get_top10_by_score(train_df_full)
val_top10 = get_top10_by_score(val_df_full)

# Step 2: Build IDF dictionary from full train.tsv corpus
idf_dict = get_idf_dict(train_corpus["candidate"].tolist())

# Step 3: Feature list (12 handcrafted)
feature_names = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

# Step 4: Feature extraction
def extract_and_append_features(df, idf_dict):
    feats = df.apply(lambda row: extract_features(row['query'], row['candidate'], idf_dict), axis=1, result_type='expand')
    feats.columns = feature_names
    return pd.concat([df, feats], axis=1)

train_top10 = extract_and_append_features(train_top10, idf_dict)
val_top10 = extract_and_append_features(val_top10, idf_dict)

# Step 5: Assign relevance for ASC/DESC
def assign_relevance(df, ascending=True):
    df = df.sort_values(by=['query', 'score'], ascending=[True, ascending])
    df['relevance'] = df.groupby('query').cumcount(ascending=False)
    return df

# Step 6: Training Loop
for order in ["asc", "desc"]:
    print(f"\nTraining model for {order.upper()} order")

    ascending = (order == "asc")
    train_df = assign_relevance(train_top10.copy(), ascending=ascending)
    val_df = assign_relevance(val_top10.copy(), ascending=ascending)

    X_train = train_df[feature_names]
    y_train = train_df['relevance']
    group_train = train_df.groupby('query').size().to_list()

    X_val = val_df[feature_names]
    y_val = val_df['relevance']
    group_val = val_df.groupby('query').size().to_list()

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbosity': -1,
        'label_gain': list(range(max(y_train.max(), y_val.max()) + 1))
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            early_stopping(stopping_rounds=20),
            log_evaluation(period=10)
        ]
    )

    model.save_model(f"lambdamart_model_12_features_{order}.txt")
    print(f"Model saved to: lambdamart_model_12_features_{order}.txt")
