# -*- coding: utf-8 -*-
"""回测引擎。硬约束降仓：预留 top_n 个仓位，缺额保留现金。

缺失收益：
  - ret_panel 不 fillna，保留 NaN
  - 旧持仓收益缺失：strict_missing=True 时 raise；False 时记录到 missing_returns
      * 调仓日旧持仓：reason='zero'，按 0 处理
      * 非调仓日旧持仓：reason='filled_price_ffill'，
        用 price_panel 前向填充收盘价后重算收益（不是 ffill 日收益率）
  - 新持仓收益缺失：调仓时剔除
  - missing_returns 记录 (trade_date, index_code, reason, context)
"""
from typing import Dict, List

import numpy as np
import pandas as pd


def run_backtest(holdings_by_date: Dict[pd.Timestamp, List[str]],
                 full_returns: pd.DataFrame,
                 trading_calendar: pd.DatetimeIndex,
                 cost_rate: float = 0.001,
                 top_n: int = 5,
                 gross_override: Dict[pd.Timestamp, float] = None,
                 strict_missing: bool = True,
                 price_panel: pd.DataFrame = None,
                 return_details: bool = False):
    cal = pd.DatetimeIndex(sorted(trading_calendar))
    ret_panel = full_returns.reindex(cal)

    # 价格前向填充后重算收益：缺失日收益 = 上一有效收盘到当前有效收盘的累计变动
    filled_ret = None
    if price_panel is not None:
        px = price_panel.reindex(cal).ffill()
        filled_ret = px.pct_change()

    records = []
    weight_rows = []
    cash_rows = []
    rebalance_rows = []
    missing_returns = []

    weights = pd.Series(dtype=float)
    cash_weight = 1.0

    for date in cal:
        market_ret = 0.0
        cost = 0.0
        turnover = 0.0
        traded_fraction = 0.0
        fee_fraction = 0.0

        is_rebalance = date in holdings_by_date

        if is_rebalance:
            new_holdings = list(holdings_by_date[date])

            if len(weights) > 0:
                r_today = ret_panel.loc[date, weights.index]
                if r_today.isna().any():
                    bad = r_today[r_today.isna()].index.tolist()
                    if strict_missing:
                        raise ValueError(f'{date.date()} 调仓日旧持仓收益缺失: {bad}')
                    missing_returns.extend((date, c, 'zero', 'rebalance_old') for c in bad)
                    r_today = r_today.fillna(0.0)
                market_ret = float((weights * r_today).sum())
                denom = 1.0 + market_ret
                drifted = weights * (1 + r_today) / denom
                cash_drifted = cash_weight / denom
            else:
                drifted = pd.Series(dtype=float)
                cash_drifted = cash_weight

            if new_holdings and date in ret_panel.index:
                r_new = ret_panel.loc[date]
                new_holdings = [c for c in new_holdings
                                if c in r_new.index and pd.notna(r_new[c])]

            if new_holdings:
                n = len(new_holdings)
                if gross_override is not None and date in gross_override:
                    gross = float(gross_override[date])
                else:
                    gross = min(n / top_n, 1.0)
                new_w = pd.Series(gross / n, index=new_holdings)
                new_cash = 1.0 - gross

                old_full = pd.concat([drifted, pd.Series({'__cash__': cash_drifted})])
                new_full = pd.concat([new_w, pd.Series({'__cash__': new_cash})])
                all_codes = old_full.index.union(new_full.index)
                dw_full = (new_full.reindex(all_codes).fillna(0) -
                           old_full.reindex(all_codes).fillna(0))
                turnover = float(dw_full.abs().sum() / 2.0)

                sec_codes = drifted.index.union(new_w.index)
                traded_fraction = float(
                    (new_w.reindex(sec_codes).fillna(0) -
                     drifted.reindex(sec_codes).fillna(0)).abs().sum()
                )
                fee_fraction = traded_fraction * cost_rate
                cost = (1.0 + market_ret) * fee_fraction

                rebalance_rows.append({
                    'trade_date': date,
                    'old_holdings': ','.join(drifted.index),
                    'new_holdings': ','.join(new_holdings),
                    'old_weights': ','.join(
                        f'{c}:{drifted.get(c, 0):.17g}' for c in drifted.index),
                    'new_weights': ','.join(
                        f'{c}:{new_w.get(c, 0):.17g}' for c in new_holdings),
                    'old_cash_weight': cash_drifted,
                    'new_cash_weight': new_cash,
                    'turnover': turnover,
                    'traded_fraction': traded_fraction,
                    'fee_fraction': fee_fraction,
                    'cost': cost,
                })
                weights, cash_weight = new_w, new_cash
            else:
                if len(drifted) > 0:
                    traded_fraction = float(drifted.abs().sum())
                    old_full = pd.concat([drifted, pd.Series({'__cash__': cash_drifted})])
                    new_full = pd.Series({'__cash__': 1.0})
                    all_codes = old_full.index.union(new_full.index)
                    turnover = float(
                        (new_full.reindex(all_codes).fillna(0) -
                         old_full.reindex(all_codes).fillna(0)).abs().sum() / 2.0
                    )
                    fee_fraction = traded_fraction * cost_rate
                    cost = (1.0 + market_ret) * fee_fraction
                else:
                    traded_fraction = 0.0
                    turnover = 0.0
                    fee_fraction = 0.0
                    cost = 0.0

                rebalance_rows.append({
                    'trade_date': date,
                    'old_holdings': ','.join(drifted.index),
                    'new_holdings': '',
                    'old_weights': ','.join(
                        f'{c}:{drifted.get(c, 0):.17g}' for c in drifted.index),
                    'new_weights': '',
                    'old_cash_weight': cash_drifted,
                    'new_cash_weight': 1.0,
                    'turnover': turnover,
                    'traded_fraction': traded_fraction,
                    'fee_fraction': fee_fraction,
                    'cost': cost,
                })
                weights = pd.Series(dtype=float)
                cash_weight = 1.0

        else:
            if len(weights) > 0:
                # 核心修改：如果提供了价格面板，统一使用价格重算的收益
                if filled_ret is not None:
                    r_today = filled_ret.loc[date, weights.index]
                    
                    # 记录哪些是原始数据中缺失的（供日志审计）
                    r_orig = ret_panel.loc[date, weights.index]
                    if r_orig.isna().any():
                        bad = r_orig[r_orig.isna()].index.tolist()
                        missing_returns.extend((date, c, 'filled_price_ffill', 'hold') for c in bad)
                else:
                    # 如果没有提供价格面板，退回到原来的逻辑
                    r_today = ret_panel.loc[date, weights.index]
                    if r_today.isna().any():
                        bad = r_today[r_today.isna()].index.tolist()
                        missing_returns.extend((date, c, 'filled_price_ffill', 'hold') for c in bad)
                        r_today = r_today.fillna(0.0)
                
                market_ret = float((weights * r_today).sum())
                denom = 1.0 + market_ret
                weights = weights * (1 + r_today) / denom
                cash_weight = cash_weight / denom

        net_ret = market_ret - cost
        records.append({
            'trade_date': date, 'market_ret': market_ret, 'cost': cost,
            'net_ret': net_ret, 'turnover': turnover,
            'traded_fraction': traded_fraction,
            'fee_fraction': fee_fraction,
            'n_holdings': len(weights), 'cash_weight': cash_weight,
        })

        for code, w in weights.items():
            weight_rows.append({'trade_date': date, 'index_code': code, 'weight': w})
        cash_rows.append({'trade_date': date, 'cash_weight': cash_weight})

    account = pd.DataFrame(records).set_index('trade_date')

    if missing_returns:
        print(f'  警告: 持仓收益缺失 {len(missing_returns)} 条')
        print(f'  {pd.Series([m[2] for m in missing_returns]).value_counts().to_dict()}')

    if not return_details:
        return account

    daily_weights = pd.DataFrame(weight_rows)
    cash_weights = pd.DataFrame(cash_rows)
    rebalance_detail = pd.DataFrame(rebalance_rows)
    missing_df = pd.DataFrame(missing_returns,
                              columns=['trade_date', 'index_code',
                                       'reason', 'context'])
    return account, daily_weights, cash_weights, rebalance_detail, missing_df