# test_factor.py
import pandas as pd
import numpy as np
from src.factor import compute_flow_factor


def test_not_listed_not_filled_zero():
    """未上市指数不应被填 0 参与排名。"""
    dates = pd.bdate_range('2024-01-01', periods=20)
    flow = pd.DataFrame({
        'index_code': ['A'] * 10 + ['B'] * 10,
        'trade_date': list(dates[:10]) + list(dates[10:]),
        'total_flow': 1.0, 'total_size': 1e9, 'prev_size': 1e9,
        'flow_strength': 0.01, 'n_etf': 1,
    })
    fac = compute_flow_factor(flow, dates, window=5, min_periods=3)
    # B 在 dates[10] 之前不应出现
    assert fac[fac['index_code'] == 'B']['trade_date'].min() >= dates[10]