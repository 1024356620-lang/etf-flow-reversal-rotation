# -*- coding: utf-8 -*-
"""检查 flow_cum / flow_pct 是否使用未来数据。

方法：用完整数据算因子 vs 截断到 T 日算因子，比较 T 日及以前的取值。

三个条件（缺一即 FAIL，返回非零）：
  1. 每个预定截断点必须有有效比较（重叠记录 > 0）；
  2. 截断日前的 (index_code, trade_date) 集合必须一致；
  3. 因子值的缺失状态一致，且有效值差异不超过容差。

用法：python tests/check_lookahead.py
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from config import StrategyConfig
from data_loader import load_real_data
from factor import compute_index_flow, compute_flow_factor

TOL = 1e-10


def _compare(full: pd.DataFrame, cut: pd.DataFrame, cutoff: pd.Timestamp) -> list:
    """返回失败原因列表；空列表表示该截断点通过。"""
    reasons = []

    full_c = full[full['trade_date'] <= cutoff]
    cut_c = cut[cut['trade_date'] <= cutoff]

    # 条件 2：截断日前的 (index_code, trade_date) 集合必须一致
    full_keys = set(map(tuple, full_c[['index_code', 'trade_date']].values))
    cut_keys = set(map(tuple, cut_c[['index_code', 'trade_date']].values))
    if full_keys != cut_keys:
        only_full = full_keys - cut_keys
        only_cut = cut_keys - full_keys
        reasons.append(
            f'记录集合不一致: full 独有 {len(only_full)} 条、cut 独有 {len(only_cut)} 条，'
            f'示例 full={list(only_full)[:3]}, cut={list(only_cut)[:3]}')

    # 用外连接比较，避免内连接漏掉只在一边出现的记录
    merged = full_c.merge(cut_c, on=['index_code', 'trade_date'],
                          suffixes=('_full', '_cut'), how='outer',
                          indicator=True)
    if len(merged) == 0:
        reasons.append('无重叠记录，无法比较')
        return reasons

    # 条件 3a：缺失状态一致
    for col in ('flow_cum', 'flow_pct'):
        miss_full = merged[f'{col}_full'].isna()
        miss_cut = merged[f'{col}_cut'].isna()
        mismatch = (miss_full != miss_cut).sum()
        if mismatch > 0:
            reasons.append(f'{col} 缺失状态不一致: {mismatch} 条')

    # 条件 3b：有效值差异不超过容差
    for col in ('flow_cum', 'flow_pct'):
        both = merged[[f'{col}_full', f'{col}_cut']].dropna()
        if len(both) == 0:
            reasons.append(f'{col} 无有效值可比较')
            continue
        diff = (both[f'{col}_full'] - both[f'{col}_cut']).abs()
        if diff.max() > TOL:
            reasons.append(f'{col} 最大差异 {diff.max():.3e} > {TOL}')

    return reasons


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
            print('  交易日历为空，FAIL（条件 1：必须有有效比较）')
            failures.append((cutoff, ['交易日历为空，无法比较']))
            continue

        index_flow_cut = compute_index_flow(etf_cut, data.mapping)
        factor_cut = compute_flow_factor(
            index_flow_cut, cal_cut,
            config.factor_window, config.factor_min_periods)

        reasons = _compare(factor_full, factor_cut, cutoff)
        if reasons:
            for r in reasons:
                print(f'  FAIL: {r}')
            failures.append((cutoff, reasons))
        else:
            merged = factor_full.merge(
                factor_cut, on=['index_code', 'trade_date'],
                suffixes=('_full', '_cut'), how='inner')
            print(f'  PASS（重叠 {len(merged)} 条，集合一致，缺失状态一致，差异 <= {TOL}）')

    print('\n结论:', 'PASS' if not failures else f'FAIL（{len(failures)} 个截断点）')
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)