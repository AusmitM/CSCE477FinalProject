import pandas as pd
import numpy as np
import pickle
import os

os.makedirs('baselines', exist_ok=True)

# Signal configurations: {signal_key: (filename, value_columns_to_baseline)}
# (filename, value columns to baseline)
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

def build_baseline(df, value_cols, train_frac=0.70):
    """
    Train on benign-only rows from first train_frac of windows.
    Returns dict: {col: {mu, sigma}} plus __global__ fallback.
    """
    # Sort by window, take first train_frac
    df = df.sort_values('window').reset_index(drop=True)
    cutoff = int(df['window'].max() * train_frac)
    train  = df[(df['window'] <= cutoff) & (df['label'] == 0)]

    if len(train) == 0:
        print('  WARNING: no benign training rows — using all rows')
        train = df[df['label'] == 0]
    if len(train) == 0:
        print('  WARNING: no benign rows at all — using full dataset')
        train = df

    baselines = {}
    for col in value_cols:
        if col not in train.columns:
            print(f'  WARNING: column {col} not found, skipping')
            continue
        vals = train[col].replace([np.inf, -np.inf], np.nan).dropna()
        mu    = vals.mean()
        sigma = vals.std()
        if sigma == 0 or np.isnan(sigma):
            sigma = max(vals.mean() * 0.01, 1e-6)
        baselines[col] = {
            'mu':    round(float(mu), 6),
            'sigma': round(float(sigma), 6),
            'n':     len(vals)
        }
        print(f'    {col}: mu={mu:.4f}, sigma={sigma:.4f}, n={len(vals)}')

    # Global fallback using all columns combined
    baselines['__global__'] = {'mu': 0.0, 'sigma': 1.0, 'n': 0}
    return baselines

# Building and saving baselines
all_baselines = {}

for sig_key, (filepath, cols) in SIGNAL_CONFIGS.items():
    if not os.path.exists(filepath):
        print(f'SKIP: {filepath} not found')
        continue
    print(f'\nBuilding baseline for {sig_key} — {filepath}')
    df = pd.read_csv(filepath)
    df['label'] = df['label'].fillna(0).astype(int)
    bl = build_baseline(df, cols)
    all_baselines[sig_key] = bl

with open('baselines/all_baselines.pkl', 'wb') as f:
    pickle.dump(all_baselines, f)

print('\n=== BASELINES SAVED ===')
for sig_key, bl in all_baselines.items():
    cols = [k for k in bl if k != '__global__']
    print(f'{sig_key}: {len(cols)} columns baselined')