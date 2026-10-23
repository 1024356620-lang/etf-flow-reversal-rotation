# -*- coding: utf-8 -*-
import pandas as pd
from src.config import StrategyConfig
from src.pipeline import run_with_data
from tests.synthetic import make_market_data


def test_pipeline_runs_end_to_end(tmp_path):
    data = make_market_data(n_days=250, n_codes=5)
    config = StrategyConfig(start_date='2023-01-01', end_date='2023-12-31',
                            flow_filter_mode='absolute')
    result = run_with_data(config, data, out_dir=tmp_path)

    # 账户记录非空，且每日一行
    acct = result['account']
    assert len(acct) > 0
    assert acct.index.is_unique
    assert {'market_ret', 'cost', 'net_ret', 'turnover', 'n_holdings'} <= set(acct.columns)

    # 绩效含关键字段
    assert '年化收益' in result['performance']
    assert '夏普' in result['performance']

    # 诊断表有约束字段
    diag = result['diagnostics']
    assert not diag.empty
    assert 'broad_ok' in diag.columns

    # 持仓只来自 full_returns 的列（接口契约）
    for date, holdings in result['holdings'].items():
        assert set(holdings) <= set(data.full_returns.columns)