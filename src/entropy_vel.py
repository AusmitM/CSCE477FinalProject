# novel1_entropy_velocity.py
# Entropy velocity — detect attacks earlier by flagging the RATE OF CHANGE
# of anomaly scores rather than waiting for absolute threshold breach.
# Operates on signal s4 (strongest signal, file/protocol entropy proxy).

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

os.makedirs('results/figures', exist_ok=True)

# Loading scored signals
s4 = pd.read_csv('results/scores_s4.csv').sort_values('window').reset_index(drop=True)
s1 = pd.read_csv('results/scores_s1.csv').sort_values('window').reset_index(drop=True)
s6 = pd.read_csv('results/scores_s6.csv').sort_values('window').reset_index(drop=True)
merged = pd.read_csv('results/fusion_output.csv').sort_values('window').reset_index(drop=True)

y_true = merged['label'].values

print(f"Loaded s4: {len(s4)} windows, {s4['label'].sum()} attack")
print(f"Loaded s1: {len(s1)} windows, {s1['label'].sum()} attack")
print(f"Loaded s6: {len(s6)} windows, {s6['label'].sum()} attack")

# Computing velocity and acceleration for signal s4 (the strongest signal) and also s1 and s6 for comparison
def add_velocity(df, zscore_col, suffix):
    """
    First derivative (velocity) = change in z-score between consecutive windows.
    Second derivative (acceleration) = change in velocity.
    Velocity z-score = how unusual is this rate of change vs the overall velocity dist.
    """
    df = df.copy().sort_values('window').reset_index(drop=True)
    df[f'velocity_{suffix}']     = df[zscore_col].diff().fillna(0)
    df[f'accel_{suffix}']        = df[f'velocity_{suffix}'].diff().fillna(0)

    vel_std = df[f'velocity_{suffix}'].std()
    vel_mu  = df[f'velocity_{suffix}'].mean()
    if vel_std == 0:
        vel_std = 1e-6

    df[f'vel_zscore_{suffix}'] = (
        (df[f'velocity_{suffix}'].abs() - abs(vel_mu)) / vel_std
    ).round(4)

    # Velocity alert: fires when velocity z-score exceeds 2.5
    # (lower than absolute threshold to catch early rising trends)
    df[f'vel_alert_{suffix}'] = (df[f'vel_zscore_{suffix}'] > 2.5).astype(int)
    return df

s4 = add_velocity(s4, 'max_zscore', 's4')
s1 = add_velocity(s1, 'max_zscore', 's1')
s6 = add_velocity(s6, 'max_zscore', 's6')

# Merging velocity-enhanced signals back into the main dataframe for fusion
vel_cols_s4 = ['window', 'velocity_s4', 'vel_zscore_s4', 'vel_alert_s4', 'accel_s4']
vel_cols_s1 = ['window', 'velocity_s1', 'vel_zscore_s1', 'vel_alert_s1']
vel_cols_s6 = ['window', 'velocity_s6', 'vel_zscore_s6', 'vel_alert_s6']

merged = merged.merge(s4[vel_cols_s4], on='window', how='left')
merged = merged.merge(s1[vel_cols_s1], on='window', how='left')
merged = merged.merge(s6[vel_cols_s6], on='window', how='left')
merged = merged.fillna(0)

# Velocity-enhanced fusion
# Original fusion: flag_count >= threshold on absolute z-scores
# Enhanced fusion: also fires when 2+ velocity signals spike simultaneously
# even if absolute thresholds not yet breached

merged['vel_flag_count'] = (
    merged['vel_alert_s4'].astype(int) +
    merged['vel_alert_s1'].astype(int) +
    merged['vel_alert_s6'].astype(int)
)

# Combined alert: original fusion OR velocity early warning
merged['enhanced_alert'] = (
    (merged['flag_count'] >= 2) |
    (merged['vel_flag_count'] >= 2)
).astype(int)

# Velocity-only alert (the novel contribution in isolation)
merged['velocity_only_alert'] = (merged['vel_flag_count'] >= 2).astype(int)

# Metric Comparison
print('\n' + '='*80)
print(f'{"Method":<35} {"P":>7} {"R":>7} {"F1":>7} {"TP":>7} {"FP":>7} {"FN":>7}')
print('='*80)

comparisons = {
    'Original fusion flag>=2':       (merged['flag_count'] >= 2).astype(int).values,
    'Velocity-only (vel>=2)':        merged['velocity_only_alert'].values,
    'Enhanced fusion (orig OR vel)': merged['enhanced_alert'].values,
}

for name, y_pred in comparisons.items():
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f'{name:<35} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {tp:>7} {fp:>7} {fn:>7}')
print('='*80)

# MTTD Comparison
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
    return round(np.mean(mttds), 3) if mttds else float('inf')

merged['orig_fusion2']  = (merged['flag_count'] >= 2).astype(int)
mttd_orig     = compute_mttd(merged, 'orig_fusion2')
mttd_vel      = compute_mttd(merged, 'velocity_only_alert')
mttd_enhanced = compute_mttd(merged, 'enhanced_alert')

print(f'\nMTTD comparison (lower = earlier detection):')
print(f'  Original fusion flag>=2:         {mttd_orig} windows')
print(f'  Velocity-only alert:             {mttd_vel} windows')
print(f'  Enhanced fusion (orig OR vel):   {mttd_enhanced} windows')

if mttd_enhanced < mttd_orig:
    improvement = round(mttd_orig - mttd_enhanced, 3)
    pct = round(100 * improvement / (mttd_orig + 1e-9), 1)
    print(f'  MTTD improvement: {improvement} windows ({pct}% faster)')
else:
    print(f'  Note: velocity did not improve MTTD on this dataset')

# Figure: velocity timeline around attack windows 
# Find the first contiguous attack episode for visualization
attack_idx   = merged[merged['label']==1].index.tolist()
episode_start = attack_idx[0]
episode_end   = min(episode_start + 60, len(merged)-1)

# Include 20 windows before for context
viz_start = max(0, episode_start - 20)
viz_data  = merged.iloc[viz_start:episode_end].copy()
viz_data['plot_idx'] = range(len(viz_data))
onset_idx = viz_data[viz_data['label']==1]['plot_idx'].iloc[0]

fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
fig.suptitle('Novel Addition 1: Entropy velocity fires before absolute threshold\n'
             'Signal s4 (file/protocol entropy proxy) around attack onset',
             fontsize=11, fontweight='bold')

# Panel 1: raw z-score with threshold line
axes[0].plot(viz_data['plot_idx'], viz_data['maxz_s4'],
             color='#1F4E79', linewidth=1.5, label='s4 z-score')
axes[0].axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='z=3 threshold')
axes[0].axvline(x=onset_idx, color='orange', linestyle='-',
                alpha=0.8, label='Attack onset')
axes[0].fill_between(viz_data['plot_idx'],
                     viz_data['anom_s4'] * viz_data['maxz_s4'].max(),
                     alpha=0.15, color='red', label='Anomaly flagged')
axes[0].set_ylabel('Max z-score', fontsize=9)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].set_title('Absolute z-score (original detection signal)', fontsize=9)

# Panel 2: velocity
axes[1].plot(viz_data['plot_idx'], viz_data['velocity_s4'],
             color='#E87B2E', linewidth=1.5, label='s4 velocity')
axes[1].axhline(y=0, color='gray', alpha=0.4)
axes[1].axvline(x=onset_idx, color='orange', linestyle='-', alpha=0.8)
axes[1].set_ylabel('Velocity (Δz/window)', fontsize=9)
axes[1].legend(fontsize=8)
axes[1].set_title('Entropy velocity — rate of change', fontsize=9)

# Panel 3: velocity z-score with lower threshold
axes[2].plot(viz_data['plot_idx'], viz_data['vel_zscore_s4'],
             color='#7030A0', linewidth=1.5, label='velocity z-score')
axes[2].axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='vel threshold=2.5')
axes[2].axvline(x=onset_idx, color='orange', linestyle='-', alpha=0.8)
axes[2].fill_between(viz_data['plot_idx'],
                     viz_data['vel_alert_s4'] * viz_data['vel_zscore_s4'].max(),
                     alpha=0.15, color='purple', label='Velocity alert')
axes[2].set_ylabel('Velocity z-score', fontsize=9)
axes[2].legend(fontsize=8)
axes[2].set_title('Velocity anomaly score — fires earlier', fontsize=9)

# Panel 4: side by side alert comparison
axes[3].fill_between(viz_data['plot_idx'],
                     viz_data['orig_fusion2'] * 0.9 + 0.1,
                     alpha=0.5, color='#1F4E79', label='Original fusion alert')
axes[3].fill_between(viz_data['plot_idx'],
                     viz_data['vel_alert_s4'] * 0.7,
                     alpha=0.5, color='#7030A0', label='Velocity alert')
axes[3].axvline(x=onset_idx, color='orange', linestyle='-',
                alpha=0.8, label='Attack onset')
axes[3].set_ylabel('Alert fired', fontsize=9)
axes[3].set_xlabel('Window index (relative)', fontsize=9)
axes[3].legend(fontsize=8)
axes[3].set_ylim(-0.1, 1.2)
axes[3].set_title('Alert comparison', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/novel1_entropy_velocity.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/novel1_entropy_velocity.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('\nFigure saved: results/figures/novel1_entropy_velocity.png')

# Figure 2: velocity z-score distributions attack vs benign
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, label) in zip(axes, [
    ('vel_zscore_s4', 'Signal s4 velocity z-score'),
    ('vel_zscore_s1', 'Signal s1 velocity z-score'),
    ('vel_zscore_s6', 'Signal s6 velocity z-score'),
]):
    benign_v = merged[merged['label']==0][col].clip(0, 15)
    attack_v = merged[merged['label']==1][col].clip(0, 15)
    ax.hist(benign_v, bins=40, alpha=0.6, color='#1F4E79',
            label='Benign', density=True)
    ax.hist(attack_v, bins=40, alpha=0.6, color='#C00000',
            label='Attack', density=True)
    ax.axvline(x=2.5, color='black', linestyle='--',
               alpha=0.7, label='Threshold=2.5')
    ax.set_xlabel('Velocity z-score')
    ax.set_ylabel('Density')
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle('Velocity z-score distributions: attack vs benign windows', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/novel1_velocity_distributions.png',
            dpi=150, bbox_inches='tight')
plt.savefig('results/figures/novel1_velocity_distributions.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print('Figure saved: results/figures/novel1_velocity_distributions.png')

merged.to_csv('results/novel1_output.csv', index=False)
print('\n=== NOVEL ADDITION 1 COMPLETE ===')