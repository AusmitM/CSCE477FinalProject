import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy
import os

os.makedirs('features', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# ── Load all CSVs and concatenate ────────────────────────────────────────────
DATA_DIR = 'RawData/MachineLearningCVE'

files = {
    'thursday_infiltration': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'thursday_web':          'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'friday_ddos':           'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'friday_portscan':       'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'monday_benign':         'Monday-WorkingHours.pcap_ISCX.csv',
}

dfs = []
for name, fname in files.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f'WARNING: {fname} not found, skipping')
        continue
    tmp = pd.read_csv(path, low_memory=False)
    tmp.columns = tmp.columns.str.strip()
    tmp['source_file'] = name
    dfs.append(tmp)
    print(f'Loaded {fname}: {len(tmp)} rows, labels: {tmp["Label"].value_counts().to_dict()}')

df = pd.concat(dfs, ignore_index=True)
print(f'\nTotal rows: {len(df)}')
print(f'Label distribution:\n{df["Label"].value_counts()}')

# ── Binary label ─────────────────────────────────────────────────────────────
df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)
print(f'\nAttack rows: {df["label_binary"].sum()}')
print(f'Benign rows: {(df["label_binary"]==0).sum()}')

# ── Clean infinite and NaN values ────────────────────────────────────────────
# CICIDS has inf values in Flow Bytes/s when flow duration is 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f'NaN counts in key columns:')
key_cols = ['Flow Bytes/s', 'Flow Packets/s', 'Total Length of Fwd Packets',
            'Packet Length Variance', 'Down/Up Ratio']
print(df[key_cols].isna().sum())

# ── Window assignment ─────────────────────────────────────────────────────────
# 50 rows per window — keeps attack rows in their own windows
ROWS_PER_WINDOW = 50
df['window'] = df.index // ROWS_PER_WINDOW

total_windows  = df['window'].nunique()
attack_windows = df[df['label_binary']==1]['window'].nunique()
print(f'\nWindows: {total_windows} total, {attack_windows} attack windows')

# ── Helper functions ──────────────────────────────────────────────────────────
def window_label(series):
    """Window is attack=1 if ANY flow in it is labeled attack."""
    return int(series.max())

def flow_entropy(series):
    series = series.dropna()
    if len(series) < 2:
        return 0.0
    counts, _ = np.histogram(series, bins=min(10, len(series)))
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(shannon_entropy(probs, base=2))

# ── Signal 1 — Byte volume and packet rate ───────────────────────────────────
print('\nExtracting Signal 1...')
sig1 = df.groupby('window').agg(
    byte_volume        = ('Total Length of Fwd Packets', 'sum'),
    bwd_byte_volume    = ('Total Length of Bwd Packets', 'sum'),
    fwd_packet_count   = ('Total Fwd Packets', 'sum'),
    flow_bytes_per_s   = ('Flow Bytes/s', 'mean'),
    flow_packets_per_s = ('Flow Packets/s', 'mean'),
    avg_packet_size    = ('Average Packet Size', 'mean'),
    label              = ('label_binary', window_label)
).reset_index()
sig1.to_csv('features/signal1_byte_volume.csv', index=False)
print(f'  Signal 1: {len(sig1)} windows, {sig1["label"].sum()} attack windows')

# ── Signal 2 — Auth attempt rate ─────────────────────────────────────────────
print('Extracting Signal 2...')
auth = df[df['Destination Port'].isin([22, 21, 23, 3389])]
if len(auth) > 0:
    sig2 = auth.groupby('window').agg(
        auth_flow_count  = ('Destination Port', 'count'),
        unique_dst_ports = ('Destination Port', 'nunique'),
        syn_flag_sum     = ('SYN Flag Count', 'sum'),
        label            = ('label_binary', window_label)
    ).reset_index()
else:
    # Fallback: SYN flood proxy for auth attempts
    sig2 = df.groupby('window').agg(
        auth_flow_count  = ('SYN Flag Count', 'sum'),
        unique_dst_ports = ('FIN Flag Count', 'sum'),
        syn_flag_sum     = ('SYN Flag Count', 'sum'),
        label            = ('label_binary', window_label)
    ).reset_index()
sig2.to_csv('features/signal2_auth_rate.csv', index=False)
print(f'  Signal 2: {len(sig2)} windows, {sig2["label"].sum()} attack windows')

# ── Signal 3 — DNS entropy proxy ─────────────────────────────────────────────
print('Extracting Signal 3...')
dns = df[df['Destination Port'] == 53]
if len(dns) > 0:
    sig3 = dns.groupby('window').agg(
        dns_flow_count   = ('Flow Packets/s', 'count'),
        dns_pkt_entropy  = ('Flow Packets/s', flow_entropy),
        dns_byte_entropy = ('Flow Bytes/s',   flow_entropy),
        dns_iat_entropy  = ('Flow IAT Mean',  flow_entropy),
        label            = ('label_binary',   window_label)
    ).reset_index()
else:
    print('  WARNING: No DNS flows found, creating empty signal3')
    sig3 = pd.DataFrame(columns=['window','dns_flow_count',
                                  'dns_pkt_entropy','dns_byte_entropy',
                                  'dns_iat_entropy','label'])
sig3.to_csv('features/signal3_dns_entropy.csv', index=False)
print(f'  Signal 3: {len(sig3)} windows, {int(sig3["label"].sum()) if len(sig3)>0 else 0} attack windows')

# ── Signal 4 — File access entropy ───────────────────────────────────────────
print('Extracting Signal 4...')
file_ports = [445, 139, 3306, 5432, 1433, 2049, 80, 443, 8080]
file_flows = df[df['Destination Port'].isin(file_ports)]
sig4 = file_flows.groupby('window').agg(
    file_flow_count    = ('Flow Packets/s', 'count'),
    pkt_len_variance   = ('Packet Length Variance', 'mean'),
    file_pkt_entropy   = ('Packet Length Variance', flow_entropy),
    fwd_bwd_ratio      = ('Down/Up Ratio', 'mean'),
    label              = ('label_binary', window_label)
).reset_index()
sig4.to_csv('features/signal4_file_entropy.csv', index=False)
print(f'  Signal 4: {len(sig4)} windows, {sig4["label"].sum()} attack windows')

# ── Signal 5 — Privilege escalation heuristic ────────────────────────────────
print('Extracting Signal 5...')
SUSP_PORTS = [4444, 1337, 31337, 9090, 5555, 6666, 7777, 8888]
privesc = df[
    df['Destination Port'].isin(SUSP_PORTS) |
    (df['URG Flag Count'] > 0) |
    (df['PSH Flag Count'] > 5) |
    (df['RST Flag Count'] > 3)
]
sig5 = privesc.groupby('window').agg(
    privesc_count = ('Destination Port', 'count'),
    urg_sum       = ('URG Flag Count', 'sum'),
    psh_sum       = ('PSH Flag Count', 'sum'),
    rst_sum       = ('RST Flag Count', 'sum'),
    label         = ('label_binary', window_label)
).reset_index()
sig5.to_csv('features/signal5_privesc.csv', index=False)
print(f'  Signal 5: {len(sig5)} windows, {sig5["label"].sum()} attack windows')

# ── Signal 6 — Flow asymmetry (IPC proxy) ────────────────────────────────────
print('Extracting Signal 6...')
df['flow_asymmetry'] = (
    (df['Total Length of Fwd Packets'] - df['Total Length of Bwd Packets']).abs() /
    (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets'] + 1)
)
sig6 = df.groupby('window').agg(
    avg_asymmetry     = ('flow_asymmetry', 'mean'),
    max_asymmetry     = ('flow_asymmetry', 'max'),
    asymmetry_entropy = ('flow_asymmetry', flow_entropy),
    down_up_ratio     = ('Down/Up Ratio', 'mean'),
    init_win_fwd      = ('Init_Win_bytes_forward', 'mean'),
    label             = ('label_binary', window_label)
).reset_index()
sig6.to_csv('features/signal6_ipc_proxy.csv', index=False)
print(f'  Signal 6: {len(sig6)} windows, {sig6["label"].sum()} attack windows')

# ── Final summary ─────────────────────────────────────────────────────────────
print('\n=== EXTRACTION COMPLETE ===')
for i, sig in enumerate([sig1,sig2,sig3,sig4,sig5,sig6], 1):
    if len(sig) > 0:
        print(f'Signal {i}: {len(sig)} windows, '
              f'{int(sig["label"].sum())} attack, '
              f'{len(sig)-int(sig["label"].sum())} benign')