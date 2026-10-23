# -*- coding: utf-8 -*-
"""数据加载与清洗。不计算因子、不筛样本池。"""
import glob
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DATA_DIR, StrategyConfig


@dataclass
class MarketData:
    etf_daily: pd.DataFrame          # etf_code, trade_date, close, shares, etf_flow, etf_size
    index_daily: pd.DataFrame        # index_code, trade_date, close, index_ret
    inst_ratio: pd.DataFrame         # etf_code, report_date, available_date, inst_ratio
    mapping: pd.DataFrame            # etf_code, index_code
    index_name: pd.DataFrame         # index_code, index_name
    benchmark: pd.DataFrame          # trade_date, index_ret
    trading_calendar: pd.DatetimeIndex  # 真实 A 股交易日
    full_returns: pd.DataFrame       # index x date 的完整收益面板（用于持仓估值）


def _clean_index_code(s: str) -> str:
    s = str(s).strip()
    if not s or s.lower() in ('nan', 'none', ''):
        return ''
    parts = s.split()
    if not parts:
        return ''
    s = parts[0]
    return re.sub(r'\.(SZ|SH|CSI|WI)$', '', s, flags=re.IGNORECASE)


def _load_trading_calendar(index_daily: pd.DataFrame) -> pd.DatetimeIndex:
    """以沪深300或上证指数行情日期作为权威交易日历。"""
    for code in ('000300', '000001', '000985'):
        sub = index_daily[index_daily['index_code'] == code]
        if len(sub) > 0:
            return pd.DatetimeIndex(sorted(sub['trade_date'].unique()))
    return pd.DatetimeIndex(sorted(index_daily['trade_date'].unique()))


def load_real_data(config: StrategyConfig) -> MarketData:
    warnings.filterwarnings('ignore', message='Workbook contains no default style')
    print('\n[数据] 读取 CSMAR 数据...')

    # ---------- ETF 基础信息 ----------
    etf_info = pd.read_excel(os.path.join(DATA_DIR, 'ETF.csv'))
    etf_info = etf_info.rename(columns={
        '证券代码': 'etf_code', '拟合指数': 'index_code', '当前规模(份)': 'shares_snapshot',
    })
    etf_info['etf_code'] = etf_info['etf_code'].astype(str).str.zfill(6)
    etf_info['index_code'] = etf_info['index_code'].apply(_clean_index_code)
    mapping = etf_info[['etf_code', 'index_code']].drop_duplicates()
    mapping = mapping[mapping['index_code'].str.len() > 0]
    index_name = (etf_info[['index_code', '证券简称']].drop_duplicates()
                  .groupby('index_code')['证券简称'].first().reset_index()
                  .rename(columns={'证券简称': 'index_name'}))

    # ---------- ETF 日行情 ----------
    etf_files = sorted(glob.glob(os.path.join(DATA_DIR, 'FUND_MKT_Quotation*.csv')))
    frames = []
    for f in etf_files:
        df = pd.read_csv(f).rename(columns={
            'TradingDate': 'trade_date', 'Symbol': 'etf_code',
            'ClosePrice': 'close', 'CirculatedShare': 'shares',
        })
        frames.append(df)
    etf_daily = pd.concat(frames, ignore_index=True)
    etf_daily['trade_date'] = pd.to_datetime(etf_daily['trade_date'])
    etf_daily['etf_code'] = etf_daily['etf_code'].astype(str).str.zfill(6)
    for c in ['close', 'shares']:
        etf_daily[c] = pd.to_numeric(etf_daily[c], errors='coerce')
    etf_daily = etf_daily[etf_daily['etf_code'].isin(mapping['etf_code'])]
    etf_daily = etf_daily.drop_duplicates(['etf_code', 'trade_date'], keep='last')
    etf_daily = etf_daily.sort_values(['etf_code', 'trade_date'])
    # 份额/价格缺失用前值填充，但保留一个标记列，便于下游判断
    etf_daily['shares_ffilled'] = etf_daily.groupby('etf_code')['shares'].transform(
        lambda s: s.ffill().notna() & s.isna())
    etf_daily['shares'] = etf_daily.groupby('etf_code')['shares'].ffill()
    etf_daily['close'] = etf_daily.groupby('etf_code')['close'].ffill()
    etf_daily = etf_daily[(etf_daily['trade_date'] >= config.start_date) &
                          (etf_daily['trade_date'] <= config.end_date)]
    etf_daily['share_chg'] = etf_daily.groupby('etf_code')['shares'].diff()
    etf_daily['etf_flow'] = etf_daily['share_chg'] * etf_daily['close']
    etf_daily['etf_size'] = etf_daily['shares'] * etf_daily['close']

    # ---------- 指数日行情 ----------
    index_daily = pd.read_csv(os.path.join(DATA_DIR, 'csmar_index_daily_clean.csv'),
                              parse_dates=['trade_date'], dtype={'index_code': str})
    index_daily['index_code'] = index_daily['index_code'].astype(str)
    index_daily = index_daily[(index_daily['trade_date'] >= config.start_date) &
                              (index_daily['trade_date'] <= config.end_date)]
    index_daily = index_daily.sort_values(['index_code', 'trade_date'])
    index_daily['index_ret'] = index_daily.groupby('index_code')['close'].pct_change()

    trading_calendar = _load_trading_calendar(index_daily)

    # 完整收益面板：index x date，用于持仓估值，绝不经过样本池过滤
    full_returns = index_daily.pivot_table(
        index='trade_date', columns='index_code', values='index_ret')
    full_returns.index.name = 'trade_date'
    full_returns.columns.name = 'index_code'
    full_returns = full_returns.reindex(trading_calendar)

    # ---------- 机构占比 ----------
    inst_ratio = pd.read_csv(os.path.join(DATA_DIR, 'Fund_HolderStructure.csv')).rename(columns={
        'Symbol': 'etf_code', 'EndDate': 'report_date', 'InstitutionRatio': 'inst_ratio',
    })
    inst_ratio['etf_code'] = inst_ratio['etf_code'].astype(str).str.zfill(6)
    inst_ratio['report_date'] = pd.to_datetime(inst_ratio['report_date'])
    inst_ratio['inst_ratio'] = pd.to_numeric(inst_ratio['inst_ratio'], errors='coerce')
    if inst_ratio['inst_ratio'].max() > 1.5:
        inst_ratio['inst_ratio'] = inst_ratio['inst_ratio'] / 100
    inst_ratio = inst_ratio[inst_ratio['etf_code'].isin(mapping['etf_code'])]
    inst_ratio = inst_ratio.drop_duplicates(['etf_code', 'report_date'], keep='last')
    inst_ratio['available_date'] = inst_ratio['report_date'] + pd.DateOffset(
        months=config.inst_lag_months)

    # ---------- 基准 ----------
    wind_path = os.path.join(DATA_DIR, 'wind_all_a.csv')
    if os.path.exists(wind_path):
        benchmark = pd.read_csv(wind_path, parse_dates=['trade_date']).sort_values('trade_date')
        benchmark['index_ret'] = benchmark['close'].pct_change()
    else:
        sub = index_daily[index_daily['index_code'] == '000985'][['trade_date', 'close']].copy()
        sub['index_ret'] = sub['close'].pct_change()
        benchmark = sub

    print(f'  交易日历: {trading_calendar.min().date()} ~ {trading_calendar.max().date()}, '
          f'{len(trading_calendar)} 天')
    print(f'  完整收益面板: {full_returns.shape}')

    return MarketData(etf_daily, index_daily, inst_ratio, mapping, index_name,
                      benchmark, trading_calendar, full_returns)