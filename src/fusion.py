import pandas as pd
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

merged = pd.read_csv('results/merged_scores.csv')

print(f'Loaded merged: {merged.shape}')
print(f'Attack windows: {int(merged["label"].sum())}')
print(f'Benign windows: {int((merged["label"]==0).sum())}')

# Only use signals with attack_mean > benign_mean 
# s2, s3, s5 are dropped — inverted or no separation
# s1, s4, s6 are the working signals
ACTIVE_SIGNALS = ['s1', 's4', 's6']
ACTIVE_ANOM    = [f'anom_{s}' for s in ACTIVE_SIGNALS]
ACTIVE_MAXZ    = [f'maxz_{s}' for s in ACTIVE_SIGNALS]

# Signal weights — proportional to z-score separation observed
# s4 has highest separation (9.887 vs 1.024), then s6, then s1
WEIGHTS = {
    'anom_s1': 0.25,
    'anom_s4': 0.50,   # strongest signal
    'anom_s6': 0.25,
}

# Confidence score
def compute_confidence(row):
    score      = sum(WEIGHTS[col] * row[col] for col in ACTIVE_ANOM)
    flag_count = int(sum(row[col] for col in ACTIVE_ANOM))
    if flag_count == 0:
        level = 'none'
    elif flag_count == 1:
        level = 'low'
    elif flag_count == 2:
        level = 'medium'
    else:
        level = 'high'
    return pd.Series({
        'confidence':    round(score, 4),
        'flag_count':    flag_count,
        'alert_level':   level,
        'alert_binary':  int(flag_count >= 2)   # high-confidence = 2+ signals
    })

merged[['confidence','flag_count','alert_level','alert_binary']] = \
    merged.apply(compute_confidence, axis=1)

merged.to_csv('results/fusion_output.csv', index=False)

print('\nAlert level distribution:')
print(merged['alert_level'].value_counts())
print(f'\nHigh-confidence alerts (flag_count >= 2): {merged["alert_binary"].sum()}')

# Per-threshold sweep to find best flag_count cutoff 
print('\nThreshold sweep (flag_count >= N):')
print(f'{"Threshold":<12} {"TP":<8} {"FP":<8} {"FN":<8} {"TN":<8} '
      f'{"Prec":<8} {"Rec":<8} {"F1":<8}')

best_f1, best_thresh = 0, 1
for thresh in [1, 2, 3]:
    y_pred = (merged['flag_count'] >= thresh).astype(int)
    y_true = merged['label']
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f'{thresh:<12} {tp:<8} {fp:<8} {fn:<8} {tn:<8} '
          f'{prec:<8.3f} {rec:<8.3f} {f1:<8.3f}')
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print(f'\nBest threshold: flag_count >= {best_thresh} (F1={best_f1:.3f})')

# Single-signal baselines for comparison
print('\nSingle-signal baselines:')
for sig in ACTIVE_SIGNALS:
    col    = f'anom_{sig}'
    y_pred = merged[col].astype(int)
    y_true = merged['label']
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f'  {sig}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} '
          f'TP={tp} FP={fp} FN={fn}')

# Correlation heatmap 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

corr = merged[ACTIVE_MAXZ].corr()
corr.columns = ['Byte Vol (s1)', 'File/Proto (s4)', 'Asymmetry (s6)']
corr.index   = ['Byte Vol (s1)', 'File/Proto (s4)', 'Asymmetry (s6)']

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Cross-signal anomaly score correlation\n(active signals only)')
plt.tight_layout()
plt.savefig('results/figures/fig1_correlation_heatmap.png', dpi=150)
plt.savefig('results/figures/fig1_correlation_heatmap.pdf', dpi=300)
plt.close()
print('\nFigure saved: results/figures/fig1_correlation_heatmap.png')

# MTTD calculation 
def compute_mttd(df, pred_col, label_col='label', window_size=50):
    """
    window_size=50 rows per window at ~avg flow rate gives approximate
    time proxy. Reports in number of windows (multiply by rows/window
    for relative comparison).
    """
    df     = df.sort_values('window').reset_index(drop=True)
    mttds  = []
    in_atk = False
    start  = None
    for i, row in df.iterrows():
        if row[label_col]==1 and not in_atk:
            in_atk, start = True, i
        if in_atk and row[pred_col]==1:
            mttds.append(i - start)
            in_atk = False
        if row[label_col]==0:
            in_atk = False
    return round(np.mean(mttds), 2) if mttds else float('inf')

merged['pred_fusion'] = (merged['flag_count'] >= best_thresh).astype(int)
merged['pred_s4']     = merged['anom_s4'].astype(int)
merged['pred_s1']     = merged['anom_s1'].astype(int)

mttd_fusion = compute_mttd(merged, 'pred_fusion')
mttd_s4     = compute_mttd(merged, 'pred_s4')
mttd_s1     = compute_mttd(merged, 'pred_s1')

print(f'\nMTTD (windows):')
print(f'  Fusion (flag>={best_thresh}): {mttd_fusion}')
print(f'  Single s4 only:            {mttd_s4}')
print(f'  Single s1 only:            {mttd_s1}')

print('\n=== FUSION COMPLETE ===')