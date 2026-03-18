"""
complete_covariate_comparison.py
Compare models with/without covariates (neutrophils, age, sex) + transcripts
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader_01 import load_and_qc_cohort

RANDOM_SEED = 42
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
# Load data & signature
# ─────────────────────────────────────────────────────────────────────────────

uganda, _ = load_and_qc_cohort('Uganda')
south,   _ = load_and_qc_cohort('South')
india,   _ = load_and_qc_cohort('India')

sig_file = OUTPUT_DIR / 'signatures' / 'uganda_optimal_13_transcripts.csv'
signature = pd.read_csv(sig_file)['transcript'].str.split('.').str[0].tolist()

print(f"Using {len(signature)}-transcript signature")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def prepare_covariates(df):
    neutrophils = df['neutrophils'].to_numpy().reshape(-1, 1)
    
    age = df['age'].to_numpy().reshape(-1, 1)
    age = (age - age.mean()) / age.std() if age.std() > 0 else age
    
    sex = df['sex'].astype(str).str.lower().str.startswith('m').astype(int).to_numpy().reshape(-1, 1)
    
    return neutrophils, age, sex


def impute_transcripts(train_df, test_df, transcripts):
    test_df = test_df.copy()
    for tx in transcripts:
        if tx not in test_df.columns:
            matches = [c for c in train_df.columns if c.startswith(tx + '.')]
            if matches and matches[0] in train_df.columns:
                mean_val = train_df[matches[0]].mean()
            elif tx in train_df.columns:
                mean_val = train_df[tx].mean()
            else:
                continue
            test_df[tx] = mean_val
    return test_df


def evaluate_models(train_df, test_df, transcripts, train_name, test_name):
    results = []
    
    y_train = (train_df['condition'] == 'ATB').astype(int).to_numpy()
    y_test  = (test_df['condition']  == 'ATB').astype(int).to_numpy()
    
    test_df_proc = impute_transcripts(train_df, test_df, transcripts)
    
    X_tx_train = train_df[transcripts].to_numpy() if all(t in train_df.columns for t in transcripts) else None
    X_tx_test  = test_df_proc[transcripts].to_numpy()
    
    neut_tr, age_tr, sex_tr = prepare_covariates(train_df)
    neut_te, age_te, sex_te = prepare_covariates(test_df)
    
    print(f"\n{train_name} → {test_name}")
    
    model_configs = [
        ('Neutrophils only',              neut_tr, neut_te),
        ('Age + Sex',                     np.hstack([age_tr, sex_tr]), np.hstack([age_te, sex_te])),
        ('All covariates',                np.hstack([neut_tr, age_tr, sex_tr]), np.hstack([neut_te, age_te, sex_te])),
        ('Transcripts only',              X_tx_train, X_tx_test),
        ('Transcripts + Neutrophils',     np.hstack([X_tx_train, neut_tr]), np.hstack([X_tx_test, neut_te])),
        ('Transcripts + All covariates',  np.hstack([X_tx_train, neut_tr, age_tr, sex_tr]), np.hstack([X_tx_test, neut_te, age_te, sex_te])),
    ]
    
    for name, X_tr, X_te in model_configs:
        if X_tr is None or X_te.shape[1] == 0:
            print(f"  {name:30s} skipped (missing data)")
            continue
            
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr_sc, y_train)
        
        y_prob = rf.predict_proba(X_te_sc)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        # simple bootstrap CI (reduced iterations for speed)
        boot_scores = []
        rng = np.random.RandomState(RANDOM_SEED)
        for _ in range(500):  # reduced from 1000 for faster run
            idx = rng.randint(0, len(y_test), len(y_test))
            if len(np.unique(y_test[idx])) < 2:
                continue
            boot_scores.append(roc_auc_score(y_test[idx], y_prob[idx]))
        
        ci_l = np.percentile(boot_scores, 2.5) if boot_scores else auc
        ci_u = np.percentile(boot_scores, 97.5) if boot_scores else auc
        
        print(f"  {name:30s} AUC {auc:.3f} (95% CI {ci_l:.3f}–{ci_u:.3f})")
        
        results.append({
            'Train': train_name,
            'Test': test_name,
            'Model': name,
            'AUC': auc,
            'CI_lower': ci_l,
            'CI_upper': ci_u
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Uganda CV (internal)
# ─────────────────────────────────────────────────────────────────────────────

print("\nUganda internal CV (10-fold)")
uganda_res = []

X_tx_ug = uganda[signature].to_numpy()
y_ug    = (uganda['condition'] == 'ATB').astype(int).to_numpy()
neut_ug, age_ug, sex_ug = prepare_covariates(uganda)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

for name, X in [
    ('Neutrophils only', neut_ug),
    ('Age + Sex', np.hstack([age_ug, sex_ug])),
    ('All covariates', np.hstack([neut_ug, age_ug, sex_ug])),
    ('Transcripts only', X_tx_ug),
    ('Transcripts + Neutrophils', np.hstack([X_tx_ug, neut_ug])),
    ('Transcripts + All covariates', np.hstack([X_tx_ug, neut_ug, age_ug, sex_ug])),
]:
    if X.shape[1] == 0:
        continue
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    rf = RandomForestClassifier(**RF_PARAMS)
    scores = cross_val_score(rf, X_sc, y_ug, cv=cv, scoring='roc_auc')
    print(f"  {name:30s} AUC {scores.mean():.3f} ± {scores.std():.3f}")
    uganda_res.append({
        'Train': 'Uganda', 'Test': 'Uganda (CV)', 'Model': name,
        'AUC': scores.mean(), 'CI_lower': scores.mean()-scores.std(), 'CI_upper': scores.mean()+scores.std()
    })

# ─────────────────────────────────────────────────────────────────────────────
# External validation
# ─────────────────────────────────────────────────────────────────────────────

south_res = evaluate_models(uganda, south, signature, 'Uganda', 'South Africa')
india_res = evaluate_models(uganda, india, signature, 'Uganda', 'India')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

all_res = uganda_res + south_res + india_res
pd.DataFrame(all_res).to_csv(
    OUTPUT_DIR / 'results' / 'covariate_comparison.csv',
    index=False
)

print("\nResults saved to VERIFICATION_OUTPUT/results/covariate_comparison.csv")