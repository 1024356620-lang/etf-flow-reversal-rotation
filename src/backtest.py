# -*- coding: utf-8 -*-
"""回测引擎。硬约束降仓：预留 top_n 个仓位，缺额保留现金。

权重口径：
  - weights 是“占当日账户总资产”的比例，sum(weights) + cash_weight = 1
  - 每日先按持仓收益更新总资产：R_t = sum(w_{t-1} * r_t)
  - 归一化：w_t = w_{t-1}(1+r_t)/(1+R_t)，cash_t = cash_{t-1}/(1+R_t)
  - 调仓日：先按旧权重算 R_t 并归一化，再设新权重
    新持仓等权，总资产仓位 gross = n/top_n，cash = 1 - gross

成本口径（统一基数）：
  - traded_fraction 是证券买卖金额 / 当日收盘扣费前净资产
  - fee_fraction = traded_fraction * cost_rate，占当日扣费前净资产
  - cost = (1 + market_ret) * fee_fraction，换算成占上日净资产
  - net_ret = market_ret - cost，三者同基数（相对上日净资产）
  - turnover 含现金，仅用于统计资产配置变化

空仓分支：
  - old_weights 写换仓前漂移权重，new_weights 为空
  - turnover 含现金，sum(|new-old|)/2，与非空仓分支一致
  - traded_fraction 只含证券，cost 只按证券计费
"""
from typing import Dict, List

import numpy as np
import pandas as pd


def run_backtest(holdings_by_date: Dict[pd.Timestamp, List[str]],
                 full_returns: pd.DataFrame,
                 trading_calendar: pd.DatetimeIndex,
                 cost_rate: float = 0.001,
                 top_n: int = 5,
                 return_details: bool = False):
    cal = pd.DatetimeIndex(sorted(trading_calendar))
    ret_panel = full_returns.reindex(cal).fillna(0.0)

    records = []
    weight_rows = []
    cash_rows = []
    rebalance_rows = []

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

            # 1) 当日收益按旧权重（含现金，现金收益 0），归一化漂移
            if len(weights) > 0:
                r_today = ret_panel.loc[date, weights.index]
                market_ret = float((weights * r_today).sum())
                denom = 1.0 + market_ret
                drifted = weights * (1 + r_today) / denom
                cash_drifted = cash_weight / denom
            else:
                drifted = pd.Series(dtype=float)
                cash_drifted = cash_weight

            if new_holdings:
                n = len(new_holdings)
                gross = min(n / top_n, 1.0)
                new_w = pd.Series(gross / n, index=new_holdings)
                new_cash = 1.0 - gross

                # 2) turnover 含现金（统计用）
                old_full = pd.concat([drifted, pd.Series({'__cash__': cash_drifted})])
                new_full = pd.concat([new_w, pd.Series({'__cash__': new_cash})])
                all_codes = old_full.index.union(new_full.index)
                dw_full = (new_full.reindex(all_codes).fillna(0) -
                           old_full.reindex(all_codes).fillna(0))
                turnover = float(dw_full.abs().sum() / 2.0)

                # 3) 成本：只对证券买卖计费，基数换算到上日净资产
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
                # 空仓：清掉全部证券，现金 100%
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
                r_today = ret_panel.loc[date, weights.index]
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
    if not return_details:
        return account

    daily_weights = pd.DataFrame(weight_rows)
    cash_weights = pd.DataFrame(cash_rows)
    rebalance_detail = pd.DataFrame(rebalance_rows)
    return account, daily_weights, cash_weights, rebalance_detail