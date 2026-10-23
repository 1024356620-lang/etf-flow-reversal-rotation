# -*- coding: utf-8 -*-
"""检查 flow_cum / flow_pct 是否使用未来数据。

方法：用完整数据算因子 vs 截断到 T 日算因子，比较 T 日及以前的取值。
若差异 > 1e-10，说明用了未来数据，脚本返回非零退出码。

用法：python tests/check_lookahead.py
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from config import StrategyConfig
from data_loader import load_real_data
from factor import compute_index_flow, compute_flow_factor


def main() -> bool:
    config = StrategyConfig(min_etf_size=5e8)
    print('加载数据...')
    data = load_real_data(config)

    print('完整数据算因子...')
    index_flow_full = compute_index_flow(data.etf_daily, data.mapping)
    factor_full = compute_flow_factor(
        index_flow_full, data.trading_calendar,
        config.factor_window, config.factor_min_periods)

    cutoffs = [
        pd.Timestamp('2020-02-14'),
        pd.Timestamp('2021-05-20'),
        pd.Timestamp('2022-08-15'),
        pd.Timestamp('2023-11-15'),
        pd.Timestamp('2025-05-15'),
    ]

    failures = []
    for cutoff in cutoffs:
        print(f'\n截断到 {cutoff.date()}...')
        etf_cut = data.etf_daily[data.etf_daily['trade_date'] <= cutoff]
        cal_cut = data.trading_calendar[data.trading_calendar <= cutoff]
        if len(cal_cut) == 0:
            print('  交易日历为空，跳过')
            continue

        index_flow_cut = compute_index_flow(etf_cut, data.mapping)
        factor_cut = compute_flow_factor(
            index_flow_cut, cal_cut,
            config.factor_window, config.factor_min_periods)

        merged = factor_full.merge(
            factor_cut, on=['index_code', 'trade_date'],
            suffixes=('_full', '_cut'), how='inner')
        merged = merged[merged['trade_date'] <= cutoff]

        if len(merged) == 0:
            print('  无重叠记录，跳过')
            continue

        diff_cum = (merged['flow_cum_full'] - merged['flow_cum_cut']).abs()
        diff_pct = (merged['flow_pct_full'] - merged['flow_pct_cut']).abs()

        print(f'  重叠记录: {len(merged)}')
        print(f'  flow_cum  最大差异: {diff_cum.max():.3e}')
        print(f'  flow_pct  最大差异: {diff_pct.max():.3e}')

        if diff_cum.max() > 1e-10 or diff_pct.max() > 1e-10:
            print('  >>> flow_cum 或 flow_pct 使用了未来数据（前视）')
            failures.append(cutoff)
        else:
            print('  flow_cum / flow_pct 无前视')

    print('\n结论:', 'PASS' if not failures else f'FAIL（{len(failures)} 个截断点）')
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)