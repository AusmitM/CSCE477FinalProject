import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    RocCurveDisplay, precision_recall_curve,
    average_precision_score
)
from pyod.models.iforest import IForest

os.makedirs('results/figures', exist_ok=True)

merged = pd.read_csv('results/fusion_output.csv')
y_true = merged['label'].values

print(f'Test set: {len(merged)} windows')
print(f'Attack:   {y_true.sum()}  ({100*y_true.mean():.1f}%)')
print(f'Benign:   {(y_true==0).sum()}  ({100*(1-y_true.mean()):.1f}%)')

# Defining methods of comparison
methods = {
    'Single-signal s1 (byte volume)':   merged['anom_s1'].values,
    'Single-signal s4 (file/protocol)': merged['anom_s4'].values,
    'Single-signal s6 (asymmetry)':     merged['anom_s6'].values,
    'Fusion flag>=1 (any signal)':       (merged['flag_count'] >= 1).astype(int).values,
    'Fusion flag>=2 (2+ signals)':       (merged['flag_count'] >= 2).astype(int).values,
    'Fusion flag>=3 (all signals)':      (merged['flag_count'] >= 3).astype(int).values,
}

# Confidence scores for ROC curves
scores = {
    'Single-signal s1':   merged['maxz_s1'].values,
    'Single-signal s4':   merged['maxz_s4'].values,
    'Single-signal s6':   merged['maxz_s6'].values,
    'Multi-signal fusion': merged['confidence'].values,
}

# Isolation Forest Baseline
print('\nFitting Isolation Forest baseline...')
feat_cols = ['maxz_s1', 'maxz_s4', 'maxz_s6']
X = merged[feat_cols].fillna(0).values

# Contamination = fraction of attack windows
contam = min(0.45, round(y_true.mean(), 2))
clf = IForest(contamination=contam, random_state=42, n_estimators=100)
clf.fit(X)
merged['iforest_score'] = clf.decision_scores_
merged['iforest_pred']  = clf.labels_

iforest_pred  = merged['iforest_pred'].values
iforest_score = merged['iforest_score'].values

methods['Isolation Forest (all 3 signals)'] = iforest_pred
scores['Isolation Forest']                   = iforest_score

# Metric Table
print('\n' + '='*85)
print(f'{"Method":<40} {"P":>6} {"R":>6} {"F1":>6} {"AUC":>6} '
      f'{"TP":>6} {"FP":>6} {"FN":>6}')
print('='*85)

results = []
for name, y_pred in methods.items():
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # ROC-AUC using confidence if available, else binary pred
    sig_key = None
    for k in scores:
        if k.lower().replace(' ','') in name.lower().replace(' ',''):
            sig_key = k
            break
    if sig_key:
        try:
            auc = roc_auc_score(y_true, scores[sig_key])
        except:
            auc = roc_auc_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_pred)

    print(f'{name:<40} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {auc:>6.3f} '
          f'{tp:>6} {fp:>6} {fn:>6}')
    results.append({'method':name,'precision':p,'recall':r,
                    'f1':f1,'auc':auc,'tp':tp,'fp':fp,'fn':fn,'tn':tn})

print('='*85)
pd.DataFrame(results).to_csv('results/evaluation_table.csv', index=False)

# MTTD
def compute_mttd(df, pred_col, label_col='label'):
    df     = df.sort_values('window').reset_index(drop=True)
    mttds  = []
    in_atk, start = False, None
    for i, row in df.iterrows():
        if row[label_col]==1 and not in_atk:
            in_atk, start = True, i
        if in_atk and row[pred_col]==1:
            mttds.append(i - start)
            in_atk = False
        if row[label_col]==0:
            in_atk = False
    return round(np.mean(mttds), 2) if mttds else float('inf')

merged['pred_fusion2'] = (merged['flag_count'] >= 2).astype(int)
merged['pred_fusion3'] = (merged['flag_count'] >= 3).astype(int)
merged['pred_fusion1'] = (merged['flag_count'] >= 1).astype(int)

print('\nMTTD (lower is better — units are windows):')
for col, label in [
    ('anom_s1',      'Single s1'),
    ('anom_s4',      'Single s4'),
    ('anom_s6',      'Single s6'),
    ('pred_fusion1', 'Fusion flag>=1'),
    ('pred_fusion2', 'Fusion flag>=2'),
    ('pred_fusion3', 'Fusion flag>=3'),
    ('iforest_pred', 'Isolation Forest'),
]:
    mttd = compute_mttd(merged, col)
    print(f'  {label:<25}: {mttd}')

# ROC Curve Figure
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#C00000', '#538135', '#7030A0', '#1F4E79']
for (name, score_vals), color in zip(scores.items(), colors):
    try:
        RocCurveDisplay.from_predictions(
            y_true, score_vals, name=name, ax=ax, color=color)
    except Exception as e:
        print(f'ROC skipped for {name}: {e}')
ax.plot([0,1],[0,1],'k--',alpha=0.3,label='Random')
ax.set_title('ROC curves — multi-signal fusion vs baselines\nCICIDS2017 dataset')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.tight_layout()
plt.savefig('results/figures/fig2_roc_curves.png', dpi=150)
plt.savefig('results/figures/fig2_roc_curves.pdf', dpi=300)
plt.close()
print('\nFigure saved: fig2_roc_curves.png')

# Precission-Recall Curve Figure
fig, ax = plt.subplots(figsize=(8, 6))
for (name, score_vals), color in zip(scores.items(), colors):
    try:
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, score_vals)
        ap = average_precision_score(y_true, score_vals)
        ax.plot(rec_vals, prec_vals, color=color,
                label=f'{name} (AP={ap:.3f})')
    except Exception as e:
        print(f'PR skipped for {name}: {e}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall curves — CICIDS2017 dataset')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('results/figures/fig3_precision_recall.png', dpi=150)
plt.savefig('results/figures/fig3_precision_recall.pdf', dpi=300)
plt.close()
print('Figure saved: fig3_precision_recall.png')

# Confusion Matrices at different fusion thresholds
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
thresh_methods = [
    ('flag>=1', (merged['flag_count']>=1).astype(int).values),
    ('flag>=2', (merged['flag_count']>=2).astype(int).values),
    ('flag>=3', (merged['flag_count']>=3).astype(int).values),
]
for ax, (name, y_pred) in zip(axes, thresh_methods):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign','Attack'],
                yticklabels=['Benign','Attack'])
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ax.set_title(f'Fusion {name}\nF1={f1:.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion matrices at different fusion thresholds', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig4_confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/fig4_confusion_matrices.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('Figure saved: fig4_confusion_matrices.png')

# Z-score distributions attack vs benign
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sig_labels = {'maxz_s1':'Byte Volume (s1)',
              'maxz_s4':'File/Protocol (s4)',
              'maxz_s6':'Asymmetry (s6)'}
for ax, (col, label) in zip(axes, sig_labels.items()):
    benign_z = merged[merged['label']==0][col].clip(0, 20)
    attack_z = merged[merged['label']==1][col].clip(0, 20)
    ax.hist(benign_z, bins=40, alpha=0.6, color='#1F4E79',
            label='Benign', density=True)
    ax.hist(attack_z, bins=40, alpha=0.6, color='#C00000',
            label='Attack', density=True)
    ax.axvline(x=3.0, color='black', linestyle='--',
               alpha=0.7, label='Threshold (z=3)')
    ax.set_xlabel('Max Z-score')
    ax.set_ylabel('Density')
    ax.set_title(label)
    ax.legend(fontsize=8)
plt.suptitle('Z-score distributions: attack vs benign windows', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig5_zscore_distributions.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/fig5_zscore_distributions.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('Figure saved: fig5_zscore_distributions.png')

merged.to_csv('results/final_output.csv', index=False)
print('\n=== EVALUATION COMPLETE ===')
print('All figures saved to results/figures/')
print('Full results saved to results/evaluation_table.csv')