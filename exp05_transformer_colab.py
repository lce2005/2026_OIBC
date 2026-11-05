"""Colab: mount Drive, then execute this file. Requires torch, pandas, pyarrow.
24-hour calendar-day encoder; split boundaries are never crossed.
Only supplied forecast/time/site features enter the encoder, never actual energy.
"""
from pathlib import Path
import copy, json, random, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path('/content/drive/MyDrive/Data/data_v0.4')
EPOCHS, BATCH_SIZE, SEED = 30, 128, 42
PREDICT_TEST = False  # Enable only when evaluation predictions are needed.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Select a GPU runtime in Colab, then rerun.')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
SAVE_DIR = Path('/content/drive/MyDrive/OIBC_experiments/exp05_transformer') / datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')

def load_frame(name):
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f'{path}: mount Drive and check the directory.')
    frame = pd.read_parquet(path)
    frame['ts'] = pd.to_datetime(frame.ts, utc=True).dt.tz_convert('Asia/Seoul')
    assert not frame.duplicated(['site_id', 'ts']).any()
    return frame

df = load_frame('train_v0.4.parquet')
required = {'split', 'use_train', 'is_suspect_zero', 'is_daylight', 'cf_target', 'energy_kwh'}
assert required <= set(df.columns), required - set(df.columns)
assert df.use_train.eq(df.is_daylight.eq(1) & df.is_suspect_zero.eq(0)).all()
periods = {'train': ('2023-06-01 01:00', '2025-03-01 01:00'),
           'dev': ('2025-03-01 01:00', '2025-05-01 01:00'),
           'holdout': ('2025-05-01 01:00', '2025-06-01 01:00')}
frames = {}
for label, (start, end) in periods.items():
    mask = df.ts.ge(pd.Timestamp(start, tz='Asia/Seoul')) & df.ts.lt(pd.Timestamp(end, tz='Asia/Seoul'))
    assert mask.equals(df['split'].eq(label)), f'split/date mismatch: {label}'
    frames[label] = df.loc[mask].copy().reset_index(drop=True)
    assert len(frames[label])
    print(label, len(frames[label]), frames[label].ts.min(), frames[label].ts.max())
del df

# Explicit feature allowlist: no target-derived quality flags or SMP inputs.
base = ['capacity_kw', 'lat', 'lon', 'wa_dist', 'wb_dist', 'hour_sin', 'hour_cos',
        'doy_sin', 'doy_cos', 'elev', 'azim', 'cosz', 'is_daylight',
        'fc_filled_a', 'fc_filled_b', 'fc_missing_a', 'fc_missing_b']
weather = [c for c in frames['train'].columns if c.startswith(('wa_', 'wb_')) and c.endswith(('_r10', '_r17'))]
FEATURES = base + sorted(weather)
fit = frames['train'].loc[frames['train'].use_train.eq(1), FEATURES].replace([np.inf, -np.inf], np.nan)
median = fit.median().fillna(0)
filled = fit.fillna(median)
mean, std = filled.mean(), filled.std().fillna(1).clip(lower=1e-6)
del fit, filled
site_map = {str(v): i+1 for i,v in enumerate(sorted(frames['train'].site_id.unique()))}

class DailyData(Dataset):
    def __init__(self, frame):
        self.frame = frame
        raw = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
        self.x = np.concatenate([((raw.fillna(median)-mean)/std).to_numpy('float32'), raw.isna().to_numpy('float32')], axis=1)
        self.groups = list(frame.groupby(['site_id', frame.ts.dt.floor('D')], sort=False).indices.values())
        self.hours = frame.ts.dt.hour.to_numpy()
        self.sites = frame.site_id.astype(str).map(site_map).fillna(0).to_numpy('int64')
        self.cap = frame.capacity_kw.to_numpy('float32')
        self.day = frame.is_daylight.to_numpy(bool)
        self.y = frame.get('cf_target', pd.Series(0., index=frame.index)).to_numpy('float32')
        self.use = frame.get('use_train', pd.Series(False, index=frame.index)).to_numpy(bool)
    def __len__(self): return len(self.groups)
    def __getitem__(self, i):
        ids = self.groups[i]; h = self.hours[ids]
        x = np.zeros((24, self.x.shape[1]), 'float32'); x[h] = self.x[ids]
        valid = np.zeros(24, bool); valid[h] = True
        row = np.full(24, -1, 'int64'); row[h] = ids
        arrays = []
        for values, dtype in [(self.y, 'float32'), (self.cap, 'float32'), (self.day, bool), (self.use, bool)]:
            a = np.zeros(24, dtype=dtype); a[h] = values[ids]; arrays.append(a)
        return x, self.sites[ids[0]], valid, row, *arrays

datasets = {k: DailyData(v) for k,v in frames.items()}
loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=k=='train', num_workers=0, pin_memory=True) for k,v in datasets.items()}

class ForecastTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2*len(FEATURES), 64)
        self.site = nn.Embedding(len(site_map)+1, 64)
        self.hour = nn.Embedding(24, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))
    def forward(self, x, site, valid):
        z = self.input(x) + self.site(site)[:,None,:] + self.hour(torch.arange(24, device=x.device))[None,:,:]
        z = self.encoder(z, src_key_padding_mask=~valid)
        return 1.2 * torch.sigmoid(self.head(z).squeeze(-1))

model = ForecastTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

@torch.no_grad()
def predict(loader):
    model.eval()
    result = np.full(len(loader.dataset.frame), np.nan)
    for x, site, valid, row, y, cap, day, use in loader:
        pred = model(x.to(DEVICE), site.to(DEVICE), valid.to(DEVICE)).cpu() * cap * day
        result[row[valid].numpy()] = pred[valid].numpy()
    assert np.isfinite(result).all() and (result >= 0).all()
    return result

def metrics(frame, pred):
    err = pred - frame.energy_kwh.to_numpy()
    return {'rows': len(frame), 'mae_kwh': float(np.abs(err).mean()),
            'rmse_kwh': float(np.sqrt(np.mean(err**2))), 'bias_kwh': float(err.mean()),
            'daylight_mae_kwh': float(np.abs(err[frame.is_daylight.eq(1)]).mean())}

SAVE_DIR.mkdir(parents=True, exist_ok=False)
best, stale, history = float('inf'), 0, []
start_time = time.time()
for epoch in range(1, EPOCHS+1):
    model.train(); total, count = 0., 0
    for x, site, valid, row, y, cap, day, use in loaders['train']:
        mask = valid & use
        if not mask.any(): continue
        x, site, valid, y, cap, mask = [t.to(DEVICE) for t in (x, site, valid, y, cap, mask)]
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, site, valid)
        loss = ((pred-y).abs()*cap)[mask].mean()
        assert torch.isfinite(loss), 'Nonfinite loss'
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        n = int(mask.sum()); total += loss.item()*n; count += n
    score = metrics(frames['dev'], predict(loaders['dev']))['mae_kwh']
    history.append({'epoch': epoch, 'train_mae_kwh': total/max(count,1), 'dev_mae_kwh': score})
    print(history[-1], flush=True)
    if score < best:
        best, stale, best_epoch = score, 0, epoch
        torch.save(model.state_dict(), SAVE_DIR/'best_model.pt')
    else:
        stale += 1
        if stale >= 5: break

model.load_state_dict(torch.load(SAVE_DIR/'best_model.pt', map_location=DEVICE, weights_only=True))
info = {'experiment': 'exp05_transformer', 'data_path': str(DATA_DIR), 'features': FEATURES,
        'seed': SEED, 'best_epoch': best_epoch, 'training_seconds': time.time()-start_time,
        'architecture': {'d_model':64, 'heads':4, 'layers':2, 'ff_dim':256, 'dropout':0.1},
        'periods': periods, 'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
        'train_rows_used': int(frames['train'].use_train.sum()),
        'note': 'Calendar-day forecast-only attention. Training use_train filter. All-row evaluation. Site MAE, not official group score.'}
for name in ['dev', 'holdout']:
    frame = frames[name]; pred = predict(loaders[name]); info[name] = metrics(frame,pred)
    out = frame[['site_id','ts','energy_kwh']].rename(columns={'energy_kwh':'actual_kwh'}).copy()
    out['pred_kwh'] = pred; out['error_kwh'] = pred-out.actual_kwh
    out.sort_values(['site_id','ts']).to_csv(SAVE_DIR/f'{name}_errors.csv', index=False)
    out[['site_id','ts','pred_kwh']].sort_values(['site_id','ts']).to_csv(SAVE_DIR/f'{name}_predictions.csv', index=False)
    print(name, info[name])
pd.DataFrame(history).to_csv(SAVE_DIR/'history.csv', index=False)
(SAVE_DIR/'preprocessing.json').write_text(json.dumps({'features':FEATURES, 'median':median.to_dict(), 'mean':mean.to_dict(), 'std':std.to_dict(), 'site_map':site_map}, indent=2), encoding='utf-8')
(SAVE_DIR/'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
if PREDICT_TEST:
    test = load_frame('test_v0.4.parquet').reset_index(drop=True)
    loader = DataLoader(DailyData(test), batch_size=BATCH_SIZE, num_workers=0)
    out = test[['site_id','ts']].copy(); out['pred_kwh'] = predict(loader)
    out.to_csv(SAVE_DIR/'test_predictions.csv', index=False)
print('Saved:', SAVE_DIR)
