# -*- coding: utf-8 -*-
"""编排：数据 -> 因子 -> 样本池 -> 选股 -> 回测 -> 绩效 -> 落盘。"""
import sys
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


def _monthly_pools(universe, factor_df, config, trading_calendar):
    cal = pd.DataFrame({'trade_date': pd.DatetimeIndex(sorted(trading_calendar))})
    cal['period'] = cal['trade_date'].dt.to_period(config.rebalance_freq)
    first_dates = cal.groupby('period')['trade_date'].min()

    df = universe.merge(
        factor_df[['index_code', 'trade_date', 'flow_cum', 'flow_pct']],
        on=['index_code', 'trade_date'], how='left').dropna(subset=['flow_cum'])

    pools = {}
    for period, date in first_dates.items():
        cross = df[df['trade_date'] == date].copy()
        if config.flow_filter_mode == 'absolute':
            pool = cross[cross['flow_cum'] <= 0]
        else:
            pool = cross[cross['flow_pct'] <= 0.5]
        pools[date] = pool
    return pools


def _export_holdings_returns(account, daily_weights, full_returns, out_dir):
    """导出“上日实际持仓 × 当日”收益，缺失保留 NaN。"""
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
    """用已加载的 MarketData 跑完整流程。

    out_dir: 输出目录。None 时默认写 BASE_DIR/results；测试应传 tmp_path，
    避免覆盖真实冻结账本。
    """
    print('\n[因子] 计算资金流因子...')
    index_flow = compute_index_flow(data.etf_daily, data.mapping)
    factor_df = compute_flow_factor(index_flow, data.trading_calendar,
                                    config.factor_window,
                                    config.factor_min_periods)

    print('[样本池] 构建选股池...')
    universe = build_universe(index_flow, data.index_daily, data.inst_ratio,
                              data.mapping, data.index_name, config)
    print(f'  日均指数数: '
          f'{universe.groupby("trade_date")["index_code"].nunique().mean():.1f}')

    print('[选股] 逐月生成持仓...')
    name_map = dict(zip(data.index_name['index_code'],
                        data.index_name['index_name']))
    pools = _monthly_pools(universe, factor_df, config, data.trading_calendar)
    holdings_by_date, diagnostics = {}, []
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
    account, daily_weights, cash_weights, rebalance_detail = run_backtest(
        holdings_by_date, data.full_returns, data.trading_calendar,
        config.cost_rate, top_n=config.top_n, return_details=True)

    print('[绩效] 评估...')
    bench = data.benchmark.set_index('trade_date')['index_ret']
    perf = evaluate(account['net_ret'], bench, config.risk_free_rate)
    yearly = yearly_performance(account['net_ret'])

    # ---------- 落盘 ----------
    out_dir = Path(out_dir) if out_dir is not None else Path(BASE_DIR) / 'results'
    out_dir.mkdir(exist_ok=True, parents=True)

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
    stacked.to_csv(out_dir / 'index_returns.csv', index=False)

    _export_holdings_returns(account, daily_weights, data.full_returns, out_dir)
    print(f'  已保存: {out_dir}')

    return {'account': account, 'performance': perf, 'yearly': yearly,
            'diagnostics': diag_df, 'holdings': holdings_by_date,
            'daily_weights': daily_weights, 'cash_weights': cash_weights,
            'rebalance_detail': rebalance_detail}


def run(config: StrategyConfig) -> dict:
    data = load_real_data(config)
    return run_with_data(config, data)