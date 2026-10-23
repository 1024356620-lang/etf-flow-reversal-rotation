# -*- coding: utf-8 -*-
"""资金流因子。不读文件、不筛样本池。"""
import numpy as np
import pandas as pd


def compute_index_flow(etf_daily: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """把 ETF 级资金流聚合到指数级。"""
    df = etf_daily.copy()
    if 'index_code' not in df.columns:
        df = df.merge(mapping[['etf_code', 'index_code']], on='etf_code', how='left')
    df = df.dropna(subset=['index_code', 'close', 'etf_size'])
    df = df[df['etf_size'] > 0]
    agg = df.groupby(['index_code', 'trade_date']).agg(
        total_flow=('etf_flow', 'sum'),
        total_size=('etf_size', 'sum'),
        n_etf=('etf_code', 'nunique'),
    ).reset_index().sort_values(['index_code', 'trade_date'])
    agg['prev_size'] = agg.groupby('index_code')['total_size'].shift(1)
    agg['flow_strength'] = agg['total_flow'] / agg['prev_size']
    agg['flow_strength'] = agg['flow_strength'].replace([np.inf, -np.inf], np.nan)
    return agg


def compute_flow_factor(index_flow: pd.DataFrame,
                        trading_calendar: pd.DatetimeIndex,
                        window: int = 10,
                        min_periods: int = 5) -> pd.DataFrame:
    """在显式交易日历上计算滚动资金流强度与截面分位。

    缺失值处理：
      - 指数在当日之前从未出现（未上市）：保持 NaN，不参与排名
      - 指数当日无 ETF 记录（真实无申赎）：flow_strength 填 0
      - 指数当日有记录但 flow_strength 为 NaN（无法计算）：保持 NaN
    """
    df = index_flow.copy()
    all_codes = df['index_code'].unique()
    full_idx = pd.MultiIndex.from_product([all_codes, trading_calendar],
                                          names=['index_code', 'trade_date'])
    df = df.set_index(['index_code', 'trade_date']).reindex(full_idx).reset_index()
    df = df.sort_values(['index_code', 'trade_date'])

    # 首次出现日期：此前视为未上市
    first_seen = df.dropna(subset=['total_size']).groupby('index_code')['trade_date'].min()
    df['first_seen'] = df['index_code'].map(first_seen)
    not_listed = df['trade_date'] < df['first_seen']

    # 真实无申赎：已上市、当日有 ETF 记录但 flow_strength 缺失 -> 0
    # 未上市 -> NaN；无 ETF 记录（total_size 缺失且已上市）-> NaN（行情缺失，不硬填 0）
    df.loc[~not_listed & df['total_size'].notna() & df['flow_strength'].isna(),
           'flow_strength'] = 0.0
    df.loc[not_listed, 'flow_strength'] = np.nan

    # 有效观测天数：只统计非 NaN 的 flow_strength
    df['valid_obs'] = df.groupby('index_code')['flow_strength'].transform(
        lambda s: s.notna().rolling(window, min_periods=1).sum())

    df['flow_cum'] = df.groupby('index_code')['flow_strength'].transform(
        lambda s: s.rolling(window, min_periods=min_periods).sum())
    df.loc[df['valid_obs'] < min_periods, 'flow_cum'] = np.nan

    # 当日截面分位：选仓使用。无前视。
    # 早期曾用 to_period('2Q') + groupby('half').rank(pct=True) 计算
    # flow_cum_std 并用于选仓，截断测试确认存在前视（同季度较早日期
    # 用到较晚日期的截面数据），已删除，不再导出。
    df['flow_pct'] = df.groupby('trade_date')['flow_cum'].rank(pct=True)

    return df[['index_code', 'trade_date', 'flow_cum',
               'flow_pct', 'valid_obs']].dropna(subset=['flow_cum'])