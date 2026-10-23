# -*- coding: utf-8 -*-
"""样本池构建。输出仅用于截面选股，持仓收益请从 index_daily 完整行情取。"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import StrategyConfig


def build_universe(index_flow: pd.DataFrame,
                   index_daily: pd.DataFrame,
                   inst_ratio: pd.DataFrame,
                   mapping: pd.DataFrame,
                   index_name: pd.DataFrame,
                   config: StrategyConfig) -> pd.DataFrame:
    """返回 index_code, trade_date, prev_size, inst_ratio, index_ret, index_name。
    注意：本表经过规模与机构占比过滤，绝不能用于持仓估值。"""
    daily = index_daily[['index_code', 'trade_date', 'index_ret']].copy()
    df = index_flow.merge(daily, on=['index_code', 'trade_date'], how='left')

    inst_index = inst_ratio.merge(mapping[['etf_code', 'index_code']], on='etf_code', how='left')
    inst_index = inst_index.dropna(subset=['index_code'])
    inst_index = (inst_index.groupby(['index_code', 'available_date'])['inst_ratio']
                  .mean().reset_index().sort_values(['index_code', 'available_date']))

    df = df.sort_values(['index_code', 'trade_date']).reset_index(drop=True)
    result = []
    for idx, g in df.groupby('index_code', sort=False):
        g = g.copy()
        sub = inst_index[inst_index['index_code'] == idx]
        if sub.empty:
            g['inst_ratio'] = np.nan
            result.append(g)
            continue
        merged = pd.merge_asof(g.sort_values('trade_date'),
                               sub.sort_values('available_date'),
                               left_on='trade_date', right_on='available_date',
                               direction='backward')
        merged['index_code'] = idx
        result.append(merged)
    df = pd.concat(result, ignore_index=True)

    df = df.dropna(subset=['index_ret', 'inst_ratio'])
    df = df[df['prev_size'] > config.min_etf_size]
    df = df[df['inst_ratio'] < config.max_inst_ratio]
    df = df.merge(index_name, on='index_code', how='left')
    return df