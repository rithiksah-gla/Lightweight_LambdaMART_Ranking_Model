# train_lambdamart (12 + DRMM-W2V)
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
#from utils import extract_features, get_idf_dict, build_w2v_model, extract_drmm_w2v_features

# Load full N=50 utilities
train_df_full = pd.read_csv("/kaggle/input/new-utilities/train_dataset_lambdamart_v01_N50.csv")
val_df_full   = pd.read_csv("/kaggle/input/new-utilities/validation_dataset_lambdamart_v01_N50.csv")

# Top-10 per query by score (same as your other runs)
def get_top10_by_score(df):
    return df.sort_values(by=["query","score"], ascending=[True, False]).groupby("query").head(10).reset_index(drop=True)

train_top10 = get_top10_by_score(train_df_full)
val_top10   = get_top10_by_score(val_df_full)

# Build IDF & Word2Vec on train.tsv corpus (candidates only)
train_raw = pd.read_csv("/kaggle/input/top-50-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])
idf_dict  = get_idf_dict(train_raw["candidate"].tolist())
w2v_model = build_w2v_model(train_raw["candidate"].tolist())

# Feature names
handcrafted_features = [
    'num_q_terms','num_q_unique','num_d_terms','num_d_unique',
    'min_q_idf','max_q_idf','sum_q_idf',
    'min_d_idf','max_d_idf','sum_d_idf',
    'overlap','bm25_score'
]
drmm_features = [f'drmm_hist_bin_{i}' for i in range(20)]
feature_names = handcrafted_features + drmm_features  # 32

# Feature extraction
def extract_all_features(row):
    q, d = row['query'], row['candidate']
    return extract_features(q, d, idf_dict) + extract_drmm_w2v_features(q, d, w2v_model)

print("Extracting features (TRAIN)...")
train_feats = train_top10.apply(extract_all_features, axis=1, result_type='expand')
train_feats.columns = feature_names
train_top10 = pd.concat([train_top10, train_feats], axis=1)

print("Extracting features (VAL)...")
val_feats = val_top10.apply(extract_all_features, axis=1, result_type='expand')
val_feats.columns = feature_names
val_top10 = pd.concat([val_top10, val_feats], axis=1)

# Assign relevance for ASC/DESC
def assign_relevance(df, ascending=True):
    df = df.sort_values(by=['query','score'], ascending=[True, ascending])
    df['relevance'] = df.groupby('query').cumcount(ascending=False)
    return df

train_asc = assign_relevance(train_top10.copy(), ascending=True)
val_asc   = assign_relevance(val_top10.copy(),   ascending=True)
train_desc= assign_relevance(train_top10.copy(), ascending=False)
val_desc  = assign_relevance(val_top10.copy(),   ascending=False)

def train_and_save_model(train_df, val_df, model_name):
    X_train = train_df[feature_names]; y_train = train_df['relevance']
    X_val   = val_df[feature_names];   y_val   = val_df['relevance']
    group_train = train_df.groupby('query').size().to_list()
    group_val   = val_df.groupby('query').size().to_list()

    dtrain = lgb.Dataset(X_train, label=y_train, group=group_train)
    dval   = lgb.Dataset(X_val,   label=y_val,   group=group_val, reference=dtrain)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1,3,5],
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbosity': -1,
        'label_gain': list(range(max(y_train.max(), y_val.max()) + 1))
    }

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtrain, dval], valid_names=["train","val"],
        callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=10)]
    )
    model.save_model(model_name)
    print(f"Saved model: {model_name}")

# Train both
train_and_save_model(train_asc,  val_asc,  "lambdamart_with_32_w2v_features_asc.txt")
train_and_save_model(train_desc, val_desc, "lambdamart_with_32_w2v_features_desc.txt")
