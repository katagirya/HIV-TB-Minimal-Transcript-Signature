"""
UGANDA_DISCOVERY_FINAL_2.py
Discovery: RFE on Uganda DETs to optimal signature
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader_01 import load_and_qc_cohort

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path('VERIFICATION_OUTPUT')
for subdir in ['signatures', 'results', 'figures', 'data']:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

uganda, _ = load_and_qc_cohort('Uganda')
south,   _ = load_and_qc_cohort('South')
india,   _ = load_and_qc_cohort('India')

# Load Uganda DETs (strip version numbers for consistency)
det_path = Path('Uganda/uganda_significant_DTE_base.csv')  # adjust if needed
uganda_dets_df = pd.read_csv(det_path)

if 'transcript' in uganda_dets_df.columns:
    det_list = [t.split('.')[0] for t in uganda_dets_df['transcript']]
else:
    det_list = [str(t).split('.')[0] for t in uganda_dets_df.iloc[:,0]]

starting_transcripts = [t for t in det_list if t in uganda.columns]

print(f"Uganda     n={len(uganda):3d}  ATB={sum(uganda['condition']=='ATB')}")
print(f"South      n={len(south):3d}  ATB={sum(south['condition']=='ATB')}")
print(f"India      n={len(india):3d}  ATB={sum(india['condition']=='ATB')}")
print(f"Starting DETs available: {len(starting_transcripts)} / {len(det_list)}")

# ─────────────────────────────────────────────────────────────────────────────
# RFE
# ─────────────────────────────────────────────────────────────────────────────

X = uganda[starting_transcripts].to_numpy()
y = (uganda['condition'] == 'ATB').astype(int).to_numpy()

print(f"\nRFE on {len(starting_transcripts)} features (10-fold CV)")

rfe_results = []
current_features = starting_transcripts.copy()

while len(current_features) >= 5:
    X_cur = uganda[current_features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cur)

    rf = RandomForestClassifier(**RF_PARAMS)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    rfe_results.append({
        'n': len(current_features),
        'auc_mean': scores.mean(),
        'auc_std':  scores.std(),
        'features': current_features.copy()
    })

    if len(current_features) == 5:
        break

    rf.fit(X_scaled, y)
    imp = rf.feature_importances_
    remove_idx = np.argmin(imp)
    del current_features[remove_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Select signature
# ─────────────────────────────────────────────────────────────────────────────

df_rfe = pd.DataFrame(rfe_results)

max_auc = df_rfe['auc_mean'].max()
practical = df_rfe[(df_rfe['n'] >= 5) & (df_rfe['n'] <= 15)]

if not practical.empty:
    best_row = practical.loc[practical['auc_mean'].idxmax()]
    opt_n = int(best_row['n'])
    opt_features = [r for r in rfe_results if r['n'] == opt_n][0]['features']
else:
    opt_n = df_rfe['n'].min()
    opt_features = df_rfe.iloc[-1]['features']

print(f"\nSelected {opt_n} transcripts  (CV AUC {best_row['auc_mean']:.3f} ± {best_row['auc_std']:.3f})")

pd.DataFrame({'transcript': opt_features}).to_csv(
    OUTPUT_DIR / 'signatures' / f'uganda_optimal_{opt_n}.csv',
    index=False
)

for i, tx in enumerate(opt_features, 1):
    print(f"  {i:2d}  {tx}")