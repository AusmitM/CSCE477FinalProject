import pandas as pd
import numpy as np
import pickle
import os

os.makedirs('results', exist_ok=True)

with open('baselines/all_baselines.pkl', 'rb') as f:
    BL = pickle.load(f)

# Load source map to get correct train/test split
source_map = pd.read_csv('features/window_source_map.csv')
monday_windows = set(source_map[source_map['is_monday'] == 1]['window'].tolist())
attack_windows = set(source_map[source_map['label'] == 1]['window'].tolist())

print(f'Monday (train) windows: {len(monday_windows)}')
print(f'Attack (test)  windows: {len(attack_windows)}')
print(f'Non-monday windows:     {len(source_map) - len(monday_windows)}')

SIGNAL_CONFIGS = {
    's1': ('features/signal1_byte_volume.csv',
           ['byte_volume', 'bwd_byte_volume', 'flow_bytes_per_s',
            'flow_packets_per_s', 'avg_packet_size']),
    's2': ('features/signal2_auth_rate.csv',
           ['auth_flow_count', 'syn_flag_sum']),
    's3': ('features/signal3_dns_entropy.csv',
           ['dns_flow_count', 'dns_pkt_entropy', 'dns_byte_entropy']),
    's4': ('features/signal4_file_entropy.csv',
           ['file_flow_count', 'pkt_len_variance', 'file_pkt_entropy']),
    's5': ('features/signal5_privesc.csv',
           ['privesc_count', 'urg_sum', 'psh_sum', 'rst_sum']),
    's6': ('features/signal6_ipc_proxy.csv',
           ['avg_asymmetry', 'asymmetry_entropy', 'down_up_ratio']),
}

ZSCORE_THRESHOLD = 3.0

def score_signal(df, value_cols, baselines, sig_key):
    df = df.copy()
    df['label'] = df['label'].fillna(0).astype(int)

    # Test set = everything that is NOT a Monday window
    test = df[~df['window'].isin(monday_windows)].copy()

    if len(test) == 0:
        print(f'  WARNING: no test rows for {sig_key}')
        return pd.DataFrame()

    z_cols = []
    for col in value_cols:
        if col not in test.columns:
            continue
        if col not in baselines:
            continue
        mu    = baselines[col]['mu']
        sigma = baselines[col]['sigma']
        if sigma == 0:
            sigma = 1e-6

        z_col = f'z_{col}'
        vals = test[col].replace([np.inf, -np.inf], np.nan).fillna(mu)
        test[z_col] = ((vals - mu) / sigma).abs().round(4)
        z_cols.append(z_col)

    if not z_cols:
        return pd.DataFrame()

    test['max_zscore']  = test[z_cols].max(axis=1).round(4)
    test['mean_zscore'] = test[z_cols].mean(axis=1).round(4)
    test['is_anomaly']  = (test['max_zscore'] > ZSCORE_THRESHOLD).astype(int)
    test['sig_key']     = sig_key

    return test[['window', 'label', 'max_zscore', 'mean_zscore',
                 'is_anomaly', 'sig_key'] + z_cols]

# Scoring signals
all_scores = {}

for sig_key, (filepath, cols) in SIGNAL_CONFIGS.items():
    if not os.path.exists(filepath):
        print(f'SKIP: {filepath}')
        continue
    df     = pd.read_csv(filepath)
    bl     = BL[sig_key]
    scored = score_signal(df, cols, bl, sig_key)

    if len(scored) == 0:
        print(f'{sig_key}: no scored rows')
        continue

    all_scores[sig_key] = scored
    scored.to_csv(f'results/scores_{sig_key}.csv', index=False)

    tp = ((scored['is_anomaly']==1) & (scored['label']==1)).sum()
    fp = ((scored['is_anomaly']==1) & (scored['label']==0)).sum()
    fn = ((scored['is_anomaly']==0) & (scored['label']==1)).sum()
    tn = ((scored['is_anomaly']==0) & (scored['label']==0)).sum()
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    print(f'{sig_key}: {len(scored)} test windows | '
          f'TP={tp} FP={fp} FN={fn} TN={tn} | '
          f'P={prec:.3f} R={rec:.3f} F1={f1:.3f} | '
          f'max_z={scored["max_zscore"].max():.2f} | '
          f'attack windows in test: {scored["label"].sum()}')

# Merging singals for fusion
print('\nMerging signals for fusion...')

base = all_scores['s1'][['window', 'label']].copy()

for sig_key, scored in all_scores.items():
    rename = {
        'max_zscore': f'maxz_{sig_key}',
        'is_anomaly': f'anom_{sig_key}'
    }
    base = base.merge(
        scored[['window', 'max_zscore', 'is_anomaly']].rename(columns=rename),
        on='window', how='left'
    )

anom_cols = [c for c in base.columns if c.startswith('anom_')]
maxz_cols = [c for c in base.columns if c.startswith('maxz_')]
base[anom_cols] = base[anom_cols].fillna(0)
base[maxz_cols] = base[maxz_cols].fillna(0)

base.to_csv('results/merged_scores.csv', index=False)

print(f'Merged shape:           {base.shape}')
print(f'Merged attack windows:  {int(base["label"].sum())}')
print(f'Merged benign windows:  {int((base["label"]==0).sum())}')
print(f'\nAnomaly flag counts per signal:')
for col in anom_cols:
    print(f'  {col}: {int(base[col].sum())} flagged')

# Sanity Check: attack z score on windows
print('\nMean max_zscore in ATTACK windows vs BENIGN windows:')
for col in maxz_cols:
    atk_mean = base[base['label']==1][col].mean()
    ben_mean = base[base['label']==0][col].mean()
    print(f'  {col}: attack={atk_mean:.3f}  benign={ben_mean:.3f}')

print('\n=== ANOMALY SCORING COMPLETE ===')