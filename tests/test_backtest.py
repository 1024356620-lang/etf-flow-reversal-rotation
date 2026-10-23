# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest
from src.backtest import run_backtest


def test_rebalance_day_market_ret_not_overwritten_by_cost():
    """期初持有 A，第二天换到 B；A 当天涨 10%，净收益应 = 10% - 成本。

    top_n=1：单资产满仓，market_ret 才能等于 A 的 10%。
    """
    dates = pd.bdate_range('2024-01-01', periods=3)
    full = pd.DataFrame(0.0, index=dates, columns=['A', 'B'])
    full.loc[dates[1], 'A'] = 0.10
    holdings = {
        dates[0]: ['A'],
        dates[1]: ['B'],
    }
    acct = run_backtest(holdings, full, dates, cost_rate=0.001, top_n=1)
    day = acct.loc[dates[1]]
    assert day['market_ret'] == pytest.approx(0.10, abs=1e-9)
    assert day['cost'] > 0
    assert day['net_ret'] == pytest.approx(0.10 - day['cost'], abs=1e-9)


def test_monthly_equal_weight_drifts_within_month():
    """两资产月初各 50%，第一天 A 翻倍，第二天 B 翻倍，期末净值应为 2.00。

    top_n=2：两资产各 50%，满仓，无现金。
    """
    dates = pd.bdate_range('2024-01-01', periods=3)
    full = pd.DataFrame(0.0, index=dates, columns=['A', 'B'])
    full.loc[dates[1], 'A'] = 1.0
    full.loc[dates[2], 'B'] = 1.0
    holdings = {dates[0]: ['A', 'B']}
    acct = run_backtest(holdings, full, dates, cost_rate=0.0, top_n=2)
    nav = (1 + acct['net_ret']).prod()
    assert nav == pytest.approx(2.0, abs=1e-9)


def test_holding_return_uses_full_returns_not_universe():
    """持仓指数某天不在样本池，但当天上涨 5%，组合应体现 +5%。

    top_n=1：单资产满仓。
    """
    dates = pd.bdate_range('2024-01-01', periods=2)
    full = pd.DataFrame(0.0, index=dates, columns=['A'])
    full.loc[dates[1], 'A'] = 0.05
    holdings = {dates[0]: ['A']}
    acct = run_backtest(holdings, full, dates, cost_rate=0.0, top_n=1)
    assert acct.loc[dates[1], 'market_ret'] == pytest.approx(0.05, abs=1e-9)