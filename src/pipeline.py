# -*- coding: utf-8 -*-
"""编排：数据 -> 因子 -> 信号时点 -> 样本池 -> 选股 -> 回测 -> 绩效 -> 落盘。

flow_filter_mode：
  - 'absolute'       复现版 A：flow_cum <= 0，缺额现金
  - 'relative'       实验 B：不设阈值，按 flow_pct 排序，缺额现金
  - 'percentile'     旧改进版：flow_pct <= 0.5
  - 'c_equal_weight' 实验 C：用 A 的总仓位，等权持有全部可交易合格候选

signal_lag：
  - 't0' 用 T 日数据，T 日收盘调仓
  - 't1' 用截至 T-1 日的数据，T 日收盘调仓（保守版）
"""
import sys
from dataclasses import replace
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from config import StrategyConfig, BASE_DIR
from data_loader import load_real_data
from factor import compute_index_flow, compute_flow_factor
from universe import build_universe
from selector import select_holdings
from backtest import run_backtest
from performance import evaluate, yearly_performance


def _apply_signal_lag(factor_df, config, trading_calendar):
    df = factor_df.copy()
    df['flow_cum_raw'] = df['flow_cum']

    if config.signal_lag == 't0':
        df['signal_date'] = df['trade_date']
        return df[['index_code', 'trade_date', 'signal_date', 'flow_cum', 
                   'flow_cum_raw', 'flow_pct', 'valid_obs']]

    cal = pd.DatetimeIndex(sorted(trading_calendar))
    cal_df = pd.DataFrame({'trade_date': cal})
    cal_df['signal_date'] = cal_df['trade_date'].shift(1)
    prev_map = dict(zip(cal_df['trade_date'], cal_df['signal_date']))
    df['signal_date'] = df['trade_date'].map(prev_map)

    sig = df[['trade_date', 'index_code', 'flow_cum_raw']].rename(
        columns={'trade_date': 'signal_date', 'flow_cum_raw': 'flow_cum_sig'})
    
    df = df.merge(sig, on=['signal_date', 'index_code'], how='left')

    # 强制今天的 flow_cum 必须等于昨天的 flow_cum_raw，保留 NaN 以便剔除
    df['flow_cum'] = df['flow_cum_sig']
    
    # 在当日全量截面上重新排名（pandas 的 rank 默认排除 NaN）
    df['flow_pct'] = df.groupby('trade_date')['flow_cum'].rank(pct=True)

    return df[['index_code', 'trade_date', 'signal_date', 'flow_cum', 
               'flow_cum_raw', 'flow_pct', 'valid_obs']]


def _signal_date(date, trading_calendar, config):
    """t1 时返回 date 的前一个交易日；t0 时返回 date。"""
    if config.signal_lag == 't0':
        return date
    prev = [d for d in trading_calendar if d < date]
    return prev[-1] if prev else pd.NaT


def _monthly_pools(universe, factor_df, config, trading_calendar):
    cal = pd.DataFrame({'trade_date': pd.DatetimeIndex(sorted(trading_calendar))})
    cal['period'] = cal['trade_date'].dt.to_period(config.rebalance_freq)
    first_dates = cal.groupby('period')['trade_date'].min()

    df = universe.merge(
        factor_df[['index_code', 'trade_date', 'flow_cum', 'flow_cum_raw', 'flow_pct']],
        on=['index_code', 'trade_date'], how='left').dropna(subset=['flow_cum'])

    pools = {}
    for period, date in first_dates.items():
        cross = df[df['trade_date'] == date].copy()
        cross['signal_date'] = _signal_date(date, trading_calendar, config)
        if config.flow_filter_mode == 'absolute':
            pool = cross[cross['flow_cum'] <= 0]
        elif config.flow_filter_mode == 'relative':
            pool = cross
        elif config.flow_filter_mode == 'percentile':
            pool = cross[cross['flow_pct'] <= 0.5]
        else:  # c_equal_weight 也用全体合格候选
            pool = cross
        pools[date] = pool
    return pools


def _c_holdings(pools_relative, pools_absolute, config, name_map, full_returns,
                min_history=20, min_valid_ratio=0.8):
    """实验 C：用 A 的总仓位，等权持有全部可交易合格候选。"""
    holdings_by_date, gross_override = {}, {}
    valid = full_returns.notna().astype(float)
    valid_ratio = valid.rolling(min_history, min_periods=1).mean()

    for date in pools_relative:
        pool_rel = pools_relative[date]
        pool_abs = pools_absolute.get(date, pd.DataFrame())
        a_holdings, _ = select_holdings(pool_abs, config, name_map, sort_col='flow_pct')
        n_a = len(a_holdings)

        if len(pool_rel) == 0:
            holdings_by_date[date] = []
            gross_override[date] = 0.0
            continue

        if date in full_returns.index:
            today_valid = set(full_returns.loc[date].dropna().index.astype(str))
        else:
            today_valid = set()

        if date in valid_ratio.index:
            ratio_today = valid_ratio.loc[date]
            enough_history = set(ratio_today[ratio_today >= min_valid_ratio]
                                 .index.astype(str))
        else:
            enough_history = set()

        candidates = []
        for c in pool_rel.sort_values('flow_pct')['index_code']:
            c = str(c)
            if c not in today_valid:
                continue
            if c not in enough_history:
                continue
            candidates.append(c)

        if len(candidates) == 0:
            holdings_by_date[date] = []
            gross_override[date] = 0.0
            continue

        holdings_by_date[date] = candidates
        gross_override[date] = min(n_a / config.top_n, 1.0)
    return holdings_by_date, gross_override


def _export_signal_panel(pools, holdings_by_date, out_dir):
    rows = []
    for date, pool in pools.items():
        selected = set(holdings_by_date.get(date, []))
        for _, r in pool.iterrows():
            rows.append({
                'trade_date': date,
                'signal_date': r.get('signal_date', pd.NaT),
                'index_code': r['index_code'],
                'flow_cum': r['flow_cum'],
                'flow_cum_raw': r.get('flow_cum_raw', np.nan),
                'flow_pct': r['flow_pct'],
                'selected': r['index_code'] in selected,
            })
    pd.DataFrame(rows).to_csv(out_dir / 'signal_panel.csv', index=False, float_format='%.17g')
    print(f'  signal_panel 已保存: {out_dir / "signal_panel.csv"}')


def _export_holdings_returns(account, daily_weights, full_returns, out_dir):
    rows = []
    prev_w = None
    for d in sorted(account.index):
        if prev_w is not None and len(prev_w) > 0:
            r_today = full_returns.loc[d] if d in full_returns.index else None
            for code in prev_w.index:
                val = np.nan
                if r_today is not None and code in r_today.index:
                    val = r_today[code]
                rows.append({'trade_date': d, 'index_code': code, 'index_ret': val})
        day_w = daily_weights[daily_weights['trade_date'] == d]
        prev_w = (day_w.set_index('index_code')['weight']
                  if len(day_w) > 0 else pd.Series(dtype=float))
    pd.DataFrame(rows).to_csv(out_dir / 'holdings_returns.csv', index=False)
    print(f'  holdings_returns 已保存: {out_dir / "holdings_returns.csv"}')


def run_with_data(config: StrategyConfig, data, out_dir=None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else Path(BASE_DIR) / 'results'
    out_dir.mkdir(exist_ok=True, parents=True)

    print('\n[因子] 计算资金流因子...')
    index_flow = compute_index_flow(data.etf_daily, data.mapping)
    factor_df = compute_flow_factor(index_flow, data.trading_calendar,
                                    config.factor_window,
                                    config.factor_min_periods)
    factor_df = _apply_signal_lag(factor_df, config, data.trading_calendar)
    print(f'  信号时点: {config.signal_lag}')

    # 全交易日 × 指数的因子面板，供 verify_signal_lag.py 核验
    factor_df[['index_code', 'trade_date', 'flow_cum', 'flow_cum_raw', 'flow_pct']] \
        .to_csv(out_dir / 'factor_panel.csv', index=False, float_format='%.17g')
    print(f'  factor_panel 已保存: {out_dir / "factor_panel.csv"}')

    print('[样本池] 构建选股池...')
    universe = build_universe(index_flow, data.index_daily, data.inst_ratio,
                              data.mapping, data.index_name, config)
    
    # 构造价格面板，供回测引擎进行缺失收益的估值
    price_panel = data.index_daily.pivot_table(
        index='trade_date', columns='index_code', values='close')
    price_panel.index.name = 'trade_date'
    price_panel.columns.name = 'index_code'
    
    print(f'  日均指数数: '
          f'{universe.groupby("trade_date")["index_code"].nunique().mean():.1f}')

    print('[选股] 逐月生成持仓...')
    name_map = dict(zip(data.index_name['index_code'],
                        data.index_name['index_name']))
    pools = _monthly_pools(universe, factor_df, config, data.trading_calendar)

    diagnostics = []
    gross_override = None

    if config.flow_filter_mode == 'c_equal_weight':
        config_abs = replace(config, flow_filter_mode='absolute')
        pools_absolute = _monthly_pools(universe, factor_df, config_abs,
                                        data.trading_calendar)
        holdings_by_date, gross_override = _c_holdings(
            pools, pools_absolute, config, name_map, data.full_returns)
        for date, pool in pools.items():
            _, diag = select_holdings(pool, config, name_map, sort_col='flow_pct')
            diag['trade_date'] = date
            diag['pool_size_before_fallback'] = len(pool)
            diag['n_selected_c'] = len(holdings_by_date[date])
            diagnostics.append(diag)
    else:
        holdings_by_date = {}
        for date, pool in pools.items():
            holdings, diag = select_holdings(pool, config, name_map,
                                             sort_col='flow_pct')
            holdings_by_date[date] = holdings
            diag['trade_date'] = date
            diag['pool_size_before_fallback'] = len(pool)
            diagnostics.append(diag)

    print(f'  宽基不足、按规则留现金的调仓次数: '
          f'{sum(1 for d in diagnostics if not d["broad_ok"])}')

    print('[回测] 运行回测引擎...')
    strict = config.flow_filter_mode != 'c_equal_weight'
    
    # 接收 6 个返回值（增加了 actual_ret_df）
    account, daily_weights, cash_weights, rebalance_detail, missing_df, actual_ret_df = run_backtest(
        holdings_by_date, data.full_returns, data.trading_calendar,
        config.cost_rate, top_n=config.top_n,
        gross_override=gross_override,
        strict_missing=strict,
        price_panel=price_panel,
        return_details=True)

    print('[绩效] 评估...')
    bench = data.benchmark.set_index('trade_date')['index_ret']
    perf = evaluate(account['net_ret'], bench, config.risk_free_rate)
    yearly = yearly_performance(account['net_ret'])

    account.to_csv(out_dir / 'account.csv')
    daily_weights.to_csv(out_dir / 'daily_weights.csv', index=False)
    cash_weights.to_csv(out_dir / 'cash_weights.csv', index=False)
    rebalance_detail.to_csv(out_dir / 'rebalance_detail.csv', index=False)

    pd.DataFrame([
        {'trade_date': d, 'holdings': ','.join(h)}
        for d, h in holdings_by_date.items()
    ]).to_csv(out_dir / 'holdings.csv', index=False)

    diag_df = pd.DataFrame(diagnostics)
    if not diag_df.empty:
        diag_df.to_csv(out_dir / 'diagnostics.csv', index=False)

    stacked = data.full_returns.stack().rename('index_ret').reset_index()
    stacked.columns = ['trade_date', 'index_code', 'index_ret']
    stacked.to_csv(out_dir / 'index_returns.csv', index=False, float_format='%.17g')

    if not missing_df.empty:
        missing_df.to_csv(out_dir / 'missing_returns.csv', index=False)
        print(f'  missing_returns 已保存: {out_dir / "missing_returns.csv"}（{len(missing_df)} 条）')


    # 保存审计追踪：实际记账的每日资产收益
    if not actual_ret_df.empty:
        actual_ret_df.to_csv(out_dir / 'actual_returns.csv', index=False, float_format='%.17g')
        print(f'  actual_returns 已保存: {out_dir / "actual_returns.csv"}（{len(actual_ret_df)} 条）')
    _export_signal_panel(pools, holdings_by_date, out_dir)
    _export_holdings_returns(account, daily_weights, data.full_returns, out_dir)
    print(f'  已保存: {out_dir}')

    return {'account': account, 'performance': perf, 'yearly': yearly,
            'diagnostics': diag_df, 'holdings': holdings_by_date,
            'daily_weights': daily_weights, 'cash_weights': cash_weights,
            'rebalance_detail': rebalance_detail,
            'missing_returns': missing_df}


def run(config: StrategyConfig, out_dir=None) -> dict:
    data = load_real_data(config)
    return run_with_data(config, data, out_dir=out_dir)