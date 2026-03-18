"""
VALIDATE_SIGNATURE.py
Validate fixed signature on South Africa and India
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader_01 import load_and_qc_cohort

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path('VERIFICATION_OUTPUT')
(OUTPUT_DIR / 'results').mkdir(parents=True, exist_ok=True)

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

uganda, _ = load_and_qc_cohort('Uganda')
south,   _ = load_and_qc_cohort('South')
india,   _ = load_and_qc_cohort('India')

sig_path = OUTPUT_DIR / 'signatures' / 'uganda_optimal_13_transcripts.csv'
signature = pd.read_csv(sig_path)['transcript'].str.split('.').str[0].tolist()

print(f"Validating {len(signature)}-transcript signature")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(df, wanted):
    mapping = {}
    for tx in wanted:
        if tx in df.columns:
            mapping[tx] = tx
        else:
            matches = [c for c in df.columns if c.startswith(tx + '.')]
            mapping[tx] = matches[0] if matches else None
    return mapping


def impute_missing(train_df, test_df, wanted, train_mapping):
    test_df = test_df.copy()
    test_mapping = get_feature_columns(test_df, wanted)
    imputed = []

    for tx in wanted:
        if test_mapping[tx] is not None:
            continue
        train_col = train_mapping[tx]
        if train_col is None:
            continue
        mean_val = train_df[train_col].mean()
        test_df[tx] = mean_val
        imputed.append(tx)

    return test_df, imputed


def validate(train_df, test_df, features, name_train, name_test):
    print(f"\n{name_train} → {name_test}")

    train_map = get_feature_columns(train_df, features)
    test_df_proc, imputed = impute_missing(train_df, test_df, features, train_map)

    used_cols = []
    for tx in features:
        c = train_map.get(tx)
        if c and tx in test_df_proc.columns:
            used_cols.append(tx)

    if not used_cols:
        print("  No features available after mapping")
        return None

    X_train = train_df[used_cols].to_numpy()
    y_train = (train_df['condition'] == 'ATB').astype(int).to_numpy()

    X_test  = test_df_proc[used_cols].to_numpy()
    y_test  = (test_df_proc['condition'] == 'ATB').astype(int).to_numpy()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_sc, y_train)

    y_prob = rf.predict_proba(X_test_sc)[:,1]
    auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, th = roc_curve(y_test, y_prob)
    youden_idx = np.argmax(tpr - fpr)
    th_y = th[youden_idx]
    y_pred = (y_prob >= th_y).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0

    print(f"  AUC:       {auc:.3f}")
    print(f"  Youden     sens/spec = {sens:.3f} / {spec:.3f}")
    print(f"  Imputed:   {len(imputed)}")

    return {
        'auc': auc, 'sens': sens, 'spec': spec,
        'n_test': len(y_test), 'n_atb': y_test.sum(),
        'imputed': len(imputed)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

res_sa   = validate(uganda, south, signature, 'Uganda', 'South Africa')
res_ind  = validate(uganda, india,  signature, 'Uganda', 'India')

# Save summary
summary = pd.DataFrame([
    {'cohort': 'South Africa', **res_sa} if res_sa else {},
    {'cohort': 'India',        **res_ind} if res_ind else {}
])
summary.to_csv(OUTPUT_DIR / 'results' / 'validation_summary.csv', index=False)