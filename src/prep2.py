"""Article 2 - causal one-minute feature construction and future-irrigation targets."""
import argparse
import numpy as np
import pandas as pd
SENSORS = ['Soil Percent','Soil Raw','Suhu Tanah','Suhu Udara','Humidity']

def parse_timestamp(series):
    s = series.astype(str)
    s = s.str.replace(
        r'^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$',
        lambda m: m.group(1) + ' ' + m.group(2).replace('-', ':'),
        regex=True,
    )
    return pd.to_datetime(s, errors='coerce')

def future_target(frame, horizon_min):
    target = np.zeros(len(frame), dtype=np.int8)
    for _, idx in frame.groupby('segment').groups.items():
        idx = np.asarray(list(idx))
        t = frame.loc[idx, 'ts'].values.astype('datetime64[ns]')
        y = frame.loc[idx, 'Label'].values
        events = t[y == 1]
        if len(events) == 0:
            continue
        pos = np.searchsorted(events, t, side='right')
        valid = pos < len(events)
        next_event = np.empty(len(t), dtype='datetime64[ns]')
        next_event[:] = np.datetime64('NaT')
        next_event[valid] = events[pos[valid]]
        delta = (next_event - t).astype('timedelta64[s]').astype(float)
        target[idx] = (valid & (delta > 0) & (delta <= horizon_min * 60)).astype(np.int8)
    return target

def main(inp, out):
    d = pd.read_csv(inp, sep=';')
    d['ts'] = parse_timestamp(d['Timestamp'])
    for c in SENSORS + ['Label']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d['Label'].notna() & d['ts'].notna()].copy()
    d = d.dropna(subset=SENSORS).sort_values('ts')
    # One-minute grid. Sensor values are means; any irrigation reading marks the minute.
    agg = {c: 'mean' for c in SENSORS}
    agg['Label'] = 'max'
    m = d.set_index('ts').groupby(pd.Grouper(freq='1min')).agg(agg)
    m = m.dropna(subset=SENSORS + ['Label']).reset_index()
    # Do not let rolling windows or future targets cross collection gaps.
    m['gap_min'] = m['ts'].diff().dt.total_seconds().div(60)
    m['segment'] = (m['gap_min'].isna() | (m['gap_min'] > 2.1)).cumsum()
    for w in (5, 15):
        g = m.groupby('segment', group_keys=False)
        for c in SENSORS:
            minp = max(2, w // 2)
            m[f'{c}_mean{w}'] = g[c].transform(lambda s: s.rolling(w, min_periods=minp).mean())
            m[f'{c}_std{w}'] = g[c].transform(lambda s: s.rolling(w, min_periods=minp).std())
    for c in SENSORS:
        m[f'{c}_diff1'] = m.groupby('segment')[c].diff(1)
        m[f'{c}_diff5'] = m.groupby('segment')[c].diff(5)
    minute_of_day = m['ts'].dt.hour * 60 + m['ts'].dt.minute
    m['sin_time'] = np.sin(2*np.pi*minute_of_day/1440)
    m['cos_time'] = np.cos(2*np.pi*minute_of_day/1440)
    for H in (10, 30, 60):
        m[f'y{H}'] = future_target(m, H)
    # Early-warning samples: the irrigation has not started at the current minute.
    m = m[m['Label'] == 0].copy()
    non_features = {'ts','Label','gap_min','segment','y10','y30','y60'}
    features = [c for c in m.columns if c not in non_features]
    m = m.dropna(subset=features).reset_index(drop=True)
    m.to_csv(out, index=False)
    print('eligible samples:', len(m), 'segments:', m['segment'].nunique(), 'features:', len(features))
    for H in (10,30,60):
        print('H=',H,'positives=',int(m[f'y{H}'].sum()))
        masks = {
            'train': m['ts'] < '2026-05-19',
            'validation': (m['ts'] >= '2026-05-19') & (m['ts'] < '2026-05-20'),
            'test': m['ts'] >= '2026-05-20',
        }
        for name, mask in masks.items():
            z=m[mask]
            print(name,len(z),int(z[f'y{H}'].sum()))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('input_csv'); ap.add_argument('--out',default='forecast_features.csv')
    a=ap.parse_args(); main(a.input_csv,a.out)
