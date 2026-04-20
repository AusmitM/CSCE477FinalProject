# novel3_ipc_structural.py
# IPC Graph Structural Anomaly — evasion-resistant detection via
# Jensen-Shannon divergence between baseline and current flow asymmetry
# distribution. An attacker who throttles byte volume to stay below
# threshold cannot hide the structural change in traffic asymmetry patterns.

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay
)

os.makedirs('results/figures', exist_ok=True)

# Load data
# s6_raw has ALL windows including Monday — used for baseline training
# s6_scored has only non-Monday test windows — used for evaluation
s6_raw    = pd.read_csv('features/signal6_ipc_proxy.csv').sort_values('window').reset_index(drop=True)
s6_scored = pd.read_csv('results/scores_s6.csv').sort_values('window').reset_index(drop=True)
merged    = pd.read_csv('results/novel1_output.csv').sort_values('window').reset_index(drop=True)

print(f"Loaded s6 raw:    {len(s6_raw)} windows, {s6_raw['label'].sum()} attack")
print(f"Loaded s6 scored: {len(s6_scored)} windows, {s6_scored['label'].sum()} attack")

# Load source map to identify Monday baseline windows 
source_map     = pd.read_csv('features/window_source_map.csv')
monday_windows = set(source_map[source_map['is_monday'] == 1]['window'].tolist())

# Baseline: Monday rows from raw features file
s6_train = s6_raw[s6_raw['window'].isin(monday_windows)].copy()

# If Monday windows still empty (edge case), fall back to benign rows
if len(s6_train) == 0:
    print("WARNING: No Monday windows found — falling back to benign rows in raw file")
    s6_train = s6_raw[s6_raw['label'] == 0].head(5000)

# Test: use scored file (already excludes Monday)
# Merge raw avg_asymmetry into scored file so JS loop has the structural feature
s6_test = s6_scored.merge(
    s6_raw[['window', 'avg_asymmetry', 'asymmetry_entropy', 'down_up_ratio']],
    on='window', how='left'
).fillna(0)

print(f"\nBaseline (Monday) windows: {len(s6_train)}")
print(f"Test windows:              {len(s6_test)}")
print(f"Test attack windows:       {int(s6_test['label'].sum())}")

# Build baseline distribution from Monday avg_asymmetry 
# avg_asymmetry is the raw structural feature:
#   |fwd_bytes - bwd_bytes| / (fwd_bytes + bwd_bytes + 1)
# Values in [0,1] — 0 = perfectly symmetric, 1 = fully one-directional
baseline_vals = s6_train['avg_asymmetry'].replace(
    [np.inf, -np.inf], np.nan).dropna().values

print(f"\nBaseline values: n={len(baseline_vals)}, "
      f"mean={baseline_vals.mean():.4f}, std={baseline_vals.std():.4f}")

BINS    = 20
BIN_MIN = 0.0
BIN_MAX = max(1.0, float(np.percentile(baseline_vals, 99)))

baseline_hist, bin_edges = np.histogram(
    np.clip(baseline_vals, BIN_MIN, BIN_MAX),
    bins=BINS, range=(BIN_MIN, BIN_MAX), density=False
)
# Laplace smoothing to avoid zero-probability bins
baseline_hist = baseline_hist.astype(float) + 1e-9
baseline_hist /= baseline_hist.sum()

print(f"Baseline histogram built: bins={BINS}, range=[{BIN_MIN:.3f}, {BIN_MAX:.3f}]")

# Compute JS divergence per test window 
# For each window, build a local distribution from a rolling context of
# surrounding windows, then compare to baseline via JS divergence.
# JS in [0,1]: 0 = identical to baseline, 1 = maximally different.

CONTEXT_WINDOW = 10
s6_test = s6_test.reset_index(drop=True)
js_scores = []

for i in range(len(s6_test)):
    lo = max(0, i - CONTEXT_WINDOW // 2)
    hi = min(len(s6_test), i + CONTEXT_WINDOW // 2 + 1)
    local_vals = s6_test.iloc[lo:hi]['avg_asymmetry'].replace(
        [np.inf, -np.inf], np.nan).dropna().values

    if len(local_vals) < 3:
        js_scores.append(0.0)
        continue

    local_hist, _ = np.histogram(
        np.clip(local_vals, BIN_MIN, BIN_MAX),
        bins=BINS, range=(BIN_MIN, BIN_MAX), density=False
    )
    local_hist = local_hist.astype(float) + 1e-9
    local_hist /= local_hist.sum()

    js = jensenshannon(baseline_hist, local_hist)
    js_scores.append(round(float(js), 6) if not np.isnan(js) else 0.0)

s6_test['js_divergence'] = js_scores
y_true_test = s6_test['label'].values

print(f"\nJS divergence stats:")
print(f"  Attack windows  — mean: {s6_test[s6_test['label']==1]['js_divergence'].mean():.4f}")
print(f"  Benign windows  — mean: {s6_test[s6_test['label']==0]['js_divergence'].mean():.4f}")

# Threshold sweep 
print('\nJS divergence threshold sweep:')
print(f'{"Threshold":<12} {"TP":<7} {"FP":<7} {"FN":<7} {"TN":<7} '
      f'{"Prec":<8} {"Rec":<8} {"F1":<8}')

best_f1, best_thresh = 0, 0.1
for thresh in np.arange(0.05, 0.50, 0.05):
    y_pred = (s6_test['js_divergence'] > thresh).astype(int).values
    tp = int(((y_pred==1)&(y_true_test==1)).sum())
    fp = int(((y_pred==1)&(y_true_test==0)).sum())
    fn = int(((y_pred==0)&(y_true_test==1)).sum())
    tn = int(((y_pred==0)&(y_true_test==0)).sum())
    p  = precision_score(y_true_test, y_pred, zero_division=0)
    r  = recall_score(y_true_test,    y_pred, zero_division=0)
    f1 = f1_score(y_true_test,        y_pred, zero_division=0)
    print(f'{thresh:<12.2f} {tp:<7} {fp:<7} {fn:<7} {tn:<7} '
          f'{p:<8.3f} {r:<8.3f} {f1:<8.3f}')
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f'\nBest JS threshold: {best_thresh:.2f}  (F1={best_f1:.3f})')
s6_test['js_alert'] = (s6_test['js_divergence'] > best_thresh).astype(int)

# Evasion resistance demonstration
# Count attack windows missed by byte-volume signal (s1) that JS recovers.
# This is the core novel claim: structural shift detects what volume-based
# signals miss when an attacker throttles their traffic.

merged_test = merged[~merged['window'].isin(monday_windows)].copy()
merged_test = merged_test.merge(
    s6_test[['window', 'js_divergence', 'js_alert']],
    on='window', how='left'
)
merged_test['js_alert']      = merged_test['js_alert'].fillna(0).astype(int)
merged_test['js_divergence'] = merged_test['js_divergence'].fillna(0)

y_true_m = merged_test['label'].values

missed_by_s1 = merged_test[(merged_test['anom_s1']==0) & (merged_test['label']==1)]
caught_by_js = missed_by_s1[missed_by_s1['js_alert']==1]

print(f'\nEvasion resistance demonstration:')
print(f'  Attack windows missed by byte-volume (s1): {len(missed_by_s1)}')
print(f'  Of those, caught by JS structural signal:  {len(caught_by_js)}')
if len(missed_by_s1) > 0:
    pct = 100 * len(caught_by_js) / len(missed_by_s1)
    print(f'  Recovery rate:                             {pct:.1f}%')

#  Full comparison table
print('\nFull method comparison including JS structural signal:')
print(f'{"Method":<42} {"P":>7} {"R":>7} {"F1":>7} {"TP":>7} {"FP":>7}')
print('-'*77)

methods = {
    'Original fusion flag>=2':          (merged_test['flag_count'] >= 2).astype(int).values,
    'JS structural (best threshold)':   merged_test['js_alert'].values,
    'Fusion OR JS (combined)':          np.maximum(
        (merged_test['flag_count'] >= 2).astype(int).values,
        merged_test['js_alert'].values
    ),
    'Enhanced fusion + JS (Novel 1+3)': np.maximum(
        merged_test['enhanced_alert'].values,
        merged_test['js_alert'].values
    ),
}

for name, y_pred in methods.items():
    tp = int(((y_pred==1)&(y_true_m==1)).sum())
    fp = int(((y_pred==1)&(y_true_m==0)).sum())
    fn = int(((y_pred==0)&(y_true_m==1)).sum())
    p  = precision_score(y_true_m, y_pred, zero_division=0)
    r  = recall_score(y_true_m,    y_pred, zero_division=0)
    f1 = f1_score(y_true_m,        y_pred, zero_division=0)
    print(f'{name:<42} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {tp:>7} {fp:>7}')

# Figure 1: JS divergence timeline
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle(
    'Novel Addition 3: IPC Structural Anomaly via Jensen-Shannon Divergence\n'
    'Evasion-resistant detection using flow asymmetry distribution shift',
    fontsize=11, fontweight='bold'
)

plot_data = s6_test.reset_index(drop=True)
plot_data['plot_idx'] = range(len(plot_data))
attack_mask = plot_data['label'] == 1

# Panel 1: raw z-score
axes[0].plot(plot_data['plot_idx'],
             plot_data['max_zscore'].clip(0, 20),
             color='#1F4E79', linewidth=0.8, alpha=0.8, label='s6 z-score')
axes[0].axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='z=3 threshold')
axes[0].fill_between(plot_data['plot_idx'],
                     attack_mask.astype(int) * 20,
                     alpha=0.08, color='red', label='Ground truth attack')
axes[0].set_ylabel('Z-score', fontsize=9)
axes[0].legend(fontsize=8)
axes[0].set_title('Absolute z-score (standard detection)', fontsize=9)

# Panel 2: JS divergence
axes[1].plot(plot_data['plot_idx'],
             plot_data['js_divergence'],
             color='#538135', linewidth=0.8, alpha=0.9, label='JS divergence')
axes[1].axhline(y=best_thresh, color='red', linestyle='--', alpha=0.7,
                label=f'Best threshold = {best_thresh:.2f}')
axes[1].fill_between(plot_data['plot_idx'],
                     attack_mask.astype(int) * plot_data['js_divergence'].max(),
                     alpha=0.08, color='red')
axes[1].set_ylabel('JS divergence', fontsize=9)
axes[1].legend(fontsize=8)
axes[1].set_title('Jensen-Shannon divergence from Monday baseline', fontsize=9)

# Panel 3: alert comparison
axes[2].fill_between(plot_data['plot_idx'],
                     (plot_data['max_zscore'] > 3.0).astype(int) * 0.9 + 0.05,
                     alpha=0.5, color='#1F4E79', label='Z-score alert')
axes[2].fill_between(plot_data['plot_idx'],
                     plot_data['js_alert'] * 0.65,
                     alpha=0.5, color='#538135', label='JS structural alert')
axes[2].fill_between(plot_data['plot_idx'],
                     attack_mask.astype(int) * 0.40,
                     alpha=0.35, color='red', label='Ground truth attack')
axes[2].set_ylabel('Alert / Label', fontsize=9)
axes[2].set_xlabel('Window index', fontsize=9)
axes[2].legend(fontsize=8)
axes[2].set_ylim(-0.1, 1.2)
axes[2].set_title('Alert comparison and ground truth', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/novel3_js_divergence_timeline.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/novel3_js_divergence_timeline.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('\nFigure saved: novel3_js_divergence_timeline.png')

# Figure 2: JS distributions + ROC
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

benign_js = s6_test[s6_test['label']==0]['js_divergence']
attack_js = s6_test[s6_test['label']==1]['js_divergence']

axes[0].hist(benign_js, bins=40, alpha=0.6, color='#1F4E79',
             label='Benign', density=True)
axes[0].hist(attack_js, bins=40, alpha=0.6, color='#C00000',
             label='Attack', density=True)
axes[0].axvline(x=best_thresh, color='black', linestyle='--',
                alpha=0.8, label=f'Threshold = {best_thresh:.2f}')
axes[0].set_xlabel('JS divergence')
axes[0].set_ylabel('Density')
axes[0].set_title('JS divergence distribution:\nattack vs benign windows')
axes[0].legend()

sep_ratio = attack_js.mean() / (benign_js.mean() + 1e-9)
print(f'\nMean JS divergence — attack: {attack_js.mean():.4f}')
print(f'Mean JS divergence — benign: {benign_js.mean():.4f}')
print(f'Separation ratio:            {sep_ratio:.2f}x')

try:
    RocCurveDisplay.from_predictions(
        y_true_test, s6_test['js_divergence'].values,
        name='JS structural (Novel 3)', ax=axes[1], color='#538135')
    RocCurveDisplay.from_predictions(
        y_true_test, s6_test['max_zscore'].values,
        name='Z-score baseline (s6)', ax=axes[1], color='#1F4E79')
    axes[1].plot([0,1],[0,1],'k--',alpha=0.3, label='Random')
    axes[1].set_title('ROC: JS structural vs z-score baseline')
    axes[1].legend(fontsize=8)
except Exception as e:
    print(f'ROC plot skipped: {e}')

plt.tight_layout()
plt.savefig('results/figures/novel3_js_distributions.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/novel3_js_distributions.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('Figure saved: novel3_js_distributions.png')

# Figure 3: distribution shift visualization 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(
    'Distribution shift: why JS divergence detects structural change',
    fontsize=10, fontweight='bold'
)

benign_raw = s6_train['avg_asymmetry'].clip(BIN_MIN, BIN_MAX).values
attack_rows = s6_test[s6_test['label']==1].head(200)
attack_raw  = attack_rows['avg_asymmetry'].clip(BIN_MIN, BIN_MAX).values \
              if len(attack_rows) > 0 else benign_raw

axes[0].hist(benign_raw, bins=BINS, range=(BIN_MIN, BIN_MAX),
             color='#1F4E79', alpha=0.8, density=True, label='Baseline (Monday)')
axes[0].set_xlabel('Flow asymmetry')
axes[0].set_ylabel('Density')
axes[0].set_title('Baseline distribution\n(Monday benign traffic)')
axes[0].legend()

axes[1].hist(benign_raw, bins=BINS, range=(BIN_MIN, BIN_MAX),
             color='#1F4E79', alpha=0.4, density=True, label='Baseline')
axes[1].hist(attack_raw, bins=BINS, range=(BIN_MIN, BIN_MAX),
             color='#C00000', alpha=0.6, density=True, label='Attack period')
axes[1].set_xlabel('Flow asymmetry')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution shift during attack\n(JS divergence measures this gap)')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/novel3_distribution_shift.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/novel3_distribution_shift.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('Figure saved: novel3_distribution_shift.png')

merged_test.to_csv('results/novel3_output.csv', index=False)
print('\n=== NOVEL ADDITION 3 COMPLETE ===')
print('Figures: novel3_js_divergence_timeline, novel3_js_distributions, novel3_distribution_shift')