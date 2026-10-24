# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-6

def _read_str(path, **kw):
    df = pd.read_csv(path, dtype={'index_code': 'string'}, **kw)
    if 'index_code' in df.columns:
        df['index_code'] = df['index_code'].str.strip()
    for c in ('flow_cum', 'flow_cum_raw', 'flow_pct'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--t0', default='results_t0')
    parser.add_argument('--t1', default='results_t1')
    args = parser.parse_args()

    t0_dir = Path(args.t0)
    t1_dir = Path(args.t1)

    t0 = _read_str(t0_dir / 'signal_panel.csv', parse_dates=['trade_date', 'signal_date'])
    t1 = _read_str(t1_dir / 'signal_panel.csv', parse_dates=['trade_date', 'signal_date'])
    fp1 = _read_str(t1_dir / 'factor_panel.csv', parse_dates=['trade_date'])
    holdings = pd.read_csv(t1_dir / 'holdings.csv', parse_dates=['trade_date']).set_index('trade_date')['holdings']

    cal = pd.DatetimeIndex(sorted(fp1['trade_date'].unique()))
    failures = []

    # [1] t0 和 t1 的 signal_date
    print('\n[1] signal_date 检查')
    # t1 必须是前一交易日
    bad_t1 = t1[t1['signal_date'] != t1['trade_date'].map(lambda d: cal[cal < d][-1] if len(cal[cal < d]) > 0 else pd.NaT)]
    if len(bad_t1) > 0:
        print(f'  FAIL: t1 有 {len(bad_t1)} 行 signal_date 不是前一交易日')
        failures.append('t1_signal_date')
    else:
        print('  PASS: t1 signal_date 全部正确')
        
    # t0 必须等于当天
    bad_t0 = t0[t0['signal_date'] != t0['trade_date']]
    if len(bad_t0) > 0:
        print(f'  FAIL: t0 有 {len(bad_t0)} 行 signal_date 不等于 trade_date')
        failures.append('t0_signal_date')
    else:
        print('  PASS: t0 signal_date 全部等于当天')

    # [2] t1 的 flow_cum 匹配
    print('\n[2] t1 的 flow_cum 与 factor_panel 在 signal_date 的 flow_cum_raw')
    # 剔除 signal_date 为 NaT 的（第一天）
    t1_valid = t1[t1['signal_date'].notna()]
    
    fp_key = fp1.set_index(['trade_date', 'index_code'])['flow_cum_raw']
    # 如果 factor_panel 有重复键，直接 FAIL
    if fp_key.index.duplicated().any():
        print('  FAIL: factor_panel 有重复的 (trade_date, index_code)')
        failures.append('dup_fp')
    else:
        diff_cum = []
        missing_cnt = 0
        for _, r in t1_valid.iterrows():
            key = (r['signal_date'], r['index_code'])
            if key not in fp_key.index:
                missing_cnt += 1
            else:
                v1, v0 = r['flow_cum'], fp_key.loc[key]
                if pd.isna(v1) or pd.isna(v0):
                    missing_cnt += 1
                else:
                    diff_cum.append(abs(v1 - v0))
        
        if missing_cnt > 0:
            print(f'  FAIL: 有 {missing_cnt} 行在 factor_panel 找不到 signal_date 原始数据或包含 NaN')
            failures.append('flow_cum_missing')
        elif max(diff_cum, default=0) > TOL:
            print(f'  FAIL: 最大差异 {max(diff_cum):.3e}')
            failures.append('flow_cum')
        else:
            print('  PASS: flow_cum 严格匹配')

    # [3] 全排名池的 flow_pct 校验
    print('\n[3] 参与排名的全体指数，其 flow_cum 必须来自 signal_date，且 flow_pct 重算一致')
    # 获取所有调仓日的全量截面
    fp_today = fp1[fp1['trade_date'].isin(t1_valid['trade_date'].unique())].copy()
    
    # 核对全量截面的 flow_cum 是否来源于 signal_date
    fp_sig_map = t1_valid[['trade_date', 'signal_date']].drop_duplicates()
    fp_today = fp_today.merge(fp_sig_map, on='trade_date', how='left')
    
    fp_sig = fp1[['trade_date', 'index_code', 'flow_cum_raw']].rename(
        columns={'trade_date': 'signal_date', 'flow_cum_raw': 'flow_cum_sig'})
    fp_today = fp_today.merge(fp_sig, on=['signal_date', 'index_code'], how='left')
    
    # 凡 flow_cum 非 NaN 的，其 flow_cum_sig 必须有值
    ranked = fp_today[fp_today['flow_cum'].notna()]
    if ranked['flow_cum_sig'].isna().any():
        n = ranked['flow_cum_sig'].isna().sum()
        print(f'  FAIL: 发现 {n} 个指数参与排名，但其 flow_cum 并非来自 signal_date')
        failures.append('pct_lookahead')
    elif (ranked['flow_cum'] - ranked['flow_cum_sig']).abs().max() > TOL:
        print('  FAIL: 全量截面的 flow_cum 不等于 signal_date 的 flow_cum_raw')
        failures.append('pct_mismatch')
    else:
        # 重算 flow_pct
        ranked['flow_pct_recon'] = ranked.groupby('trade_date')['flow_cum'].rank(pct=True)
        merged = t1_valid.merge(ranked[['trade_date', 'index_code', 'flow_pct_recon']], 
                                on=['trade_date', 'index_code'], how='left')
        
        if merged['flow_pct_recon'].isna().any():
            print('  FAIL: 部分选中的股票找不到重算的 flow_pct')
            failures.append('pct_recon_missing')
        elif (merged['flow_pct'] - merged['flow_pct_recon']).abs().max() > TOL:
            print('  FAIL: flow_pct 与全量池重算结果不一致')
            failures.append('pct_mismatch')
        else:
            print('  PASS: 全量池来源正确，flow_pct 重算一致')

    # [4] selected 与 holdings
    print('\n[4] selected 集合与 holdings.csv 对比')
    bad_sel = 0
    for d, sub in t1.groupby('trade_date'):
        sel = set(sub[sub['selected']]['index_code'])
        actual = set(str(x) for x in holdings.get(d, '').split(',') if x)
        if sel != actual:
            bad_sel += 1
    if bad_sel > 0:
        print(f'  FAIL: {bad_sel} 个调仓日不一致')
        failures.append('selected')
    else:
        print('  PASS: selected 与 holdings 一致')

    print('\n结论:', 'PASS' if not failures else f'FAIL（{failures}）')
    return 0 if not failures else 1

if __name__ == '__main__':
    raise SystemExit(main())