# -*- coding: utf-8 -*-
"""核实 missing_returns.csv 的处理方式和数据状态。

用法：python tests/check_missing_returns.py --results-dir results_C

输出：
  1. missing_returns.csv 的 reason 分布（zero / filled_ffill）
  2. 逐条查 csmar_index_daily_clean.csv，给出 data_status
     （no_record / close_nan / has_close）
  3. 从账本重建“上日权重 > 0 × 当日收益缺失”的清单，
     与 missing_returns.csv 对比，确认没有漏记
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'raw_data'


def _read_str(path, **kw):
    df = pd.read_csv(path, dtype={'index_code': 'string'}, **kw)
    if 'index_code' in df.columns:
        df['index_code'] = df['index_code'].str.strip()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results_C')
    args = parser.parse_args()
    results = Path(args.results_dir)

    m_path = results / 'missing_returns.csv'
    if not m_path.exists():
        print(f'{m_path} 不存在')
        return

    m = _read_str(m_path, parse_dates=['trade_date'])
    idx = _read_str(DATA_DIR / 'csmar_index_daily_clean.csv',
                    parse_dates=['trade_date'], low_memory=False)

    # ---------- 1. reason 分布 ----------
    print('missing_returns.csv reason 分布（处理方式）:')
    print(m['reason'].value_counts())
    print('\n按年份:')
    print(m.groupby(m['trade_date'].dt.year).size())

    # ---------- 2. 逐条 data_status ----------
    rows = []
    for _, row in m.iterrows():
        code = row['index_code']
        d = row['trade_date']
        sub = idx[(idx['index_code'] == code) & (idx['trade_date'] == d)]
        if len(sub) == 0:
            status = 'no_record'
            close = None
        elif pd.isna(sub['close'].iloc[0]):
            status = 'close_nan'
            close = None
        else:
            status = 'has_close'
            close = float(sub['close'].iloc[0])
        rows.append({'trade_date': d, 'index_code': code,
                     'reason': row['reason'], 'context': row.get('context', ''),
                     'data_status': status, 'close': close})
    check = pd.DataFrame(rows)
    print('\ndata_status 分布（原始数据状态）:')
    print(check['data_status'].value_counts())
    print('\nreason × data_status:')
    print(pd.crosstab(check['reason'], check['data_status']))
    print('\n非 has_close 的记录:')
    bad = check[check['data_status'] != 'has_close']
    print(bad.to_string(index=False))

    # ---------- 3. 从账本重建缺失清单 ----------
    acct = pd.read_csv(results / 'account.csv', parse_dates=['trade_date']).set_index('trade_date')
    dw = _read_str(results / 'daily_weights.csv', parse_dates=['trade_date'])
    ir = _read_str(results / 'index_returns.csv', parse_dates=['trade_date'])

    ret_panel = ir.pivot(index='trade_date', columns='index_code', values='index_ret')
    w_pivot = (dw.pivot(index='trade_date', columns='index_code', values='weight')
                 .fillna(0.0))
    all_dates = list(acct.index)
    w_daily = w_pivot.reindex(all_dates, fill_value=0.0)
    prev_weights = w_daily.shift(1, fill_value=0.0)

    reconstructed = []
    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pw = pw[pw > 0]
        if len(pw) == 0:
            continue
        r_today = ret_panel.loc[d] if d in ret_panel.index else pd.Series(dtype=float)
        for code, w in pw.items():
            if code not in r_today.index or not np.isfinite(r_today.get(code, np.nan)):
                reconstructed.append((d, code))
    rec_df = pd.DataFrame(reconstructed, columns=['trade_date', 'index_code'])
    print(f'\n从账本重建的缺失清单: {len(rec_df)} 条')

    m_keys = set(map(tuple, m[['trade_date', 'index_code']].values))
    rec_keys = set(map(tuple, rec_df[['trade_date', 'index_code']].values))
    only_rec = rec_keys - m_keys
    only_m = m_keys - rec_keys
    print(f'只在重建清单里（missing_returns 漏记）: {len(only_rec)} 条')
    for x in list(only_rec)[:10]:
        print(f'  {x[0].date()} {x[1]}')
    print(f'只在 missing_returns 里（重建清单没有）: {len(only_m)} 条')
    for x in list(only_m)[:10]:
        print(f'  {x[0].date()} {x[1]}')

    out = results / 'missing_returns_check.csv'
    check.to_csv(out, index=False)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()