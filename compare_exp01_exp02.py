"""Compare completed EXP-01/EXP-02 dev predictions without retraining.

Run from the 2026_OIBC project root: python compare_exp01_exp02.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path('outputs')
FILES = {
    'EXP-01 LightGBM': OUT / 'exp01_lgbm_initial_dev_errors.csv',
    'EXP-02 MLP': OUT / 'exp02_mlp_initial_dev_errors.csv',
}
frames = {}
for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; check EXP-02 output was saved')
    frame = pd.read_csv(path, usecols=['site_id', 'ts', 'actual_kwh', 'pred_kwh'])
    if frame.duplicated(['site_id', 'ts']).any():
        raise ValueError(f'Duplicate site/time keys: {path}')
    frame['ts'] = pd.to_datetime(frame['ts'], utc=True)
    if not np.isfinite(frame[['actual_kwh', 'pred_kwh']].to_numpy(dtype=np.float64)).all():
        raise ValueError(f'Non-finite values in {path}')
    frames[label] = frame

left_name, right_name = FILES
left = frames[left_name]
right = frames[right_name]
if len(left) != len(right):
    raise ValueError(f'Different row counts: {len(left)} vs {len(right)}')
matched = left.merge(right, how='inner', on=['site_id', 'ts'], suffixes=('_01', '_02'), validate='one_to_one')
if len(matched) != len(left):
    raise ValueError('Different site/time keys; comparison stopped')
if not np.allclose(matched['actual_kwh_01'], matched['actual_kwh_02'], rtol=1e-6, atol=1e-4):
    raise ValueError('Actual generation values differ; comparison stopped')

actual = matched['actual_kwh_01'].to_numpy(dtype=np.float64)
result = {'period': '2025-03-01 01:00 to 2025-05-01 00:00 Asia/Seoul', 'rows':len(matched)}
for label, col in [(left_name,'pred_kwh_01'),(right_name,'pred_kwh_02')]:
    pred = matched[col].to_numpy(dtype=np.float64)
    diff = pred - actual
    result[label] = {
        'mae_kwh': float(np.abs(diff).mean()),
        'rmse_kwh': float(np.sqrt(np.mean(diff**2))),
        'bias_kwh': float(diff.mean()),
    }

print('===== Same rows / timestamps comparison =====')
print('Matched rows:', len(matched))
for label in FILES:
    metrics = result[label]
    print(f"{label}: MAE={metrics['mae_kwh']:.4f} kWh, "
          f"RMSE={metrics['rmse_kwh']:.4f} kWh, "
          f"bias={metrics['bias_kwh']:+.4f} kWh")
mae_delta = result[right_name]['mae_kwh'] - result[left_name]['mae_kwh']
result['mlp_minus_lgbm_mae_kwh'] = mae_delta
print(f'EXP-02 minus EXP-01 MAE: {mae_delta:+.4f} kWh')
print('Note: not the competition group-level SMP-weighted penalty.')

save_path = OUT / 'exp01_vs_exp02_dev_comparison.json'
save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print('Saved:', save_path)
