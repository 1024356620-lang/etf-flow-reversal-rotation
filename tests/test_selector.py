# test_selector.py
import pandas as pd
from src.config import StrategyConfig
from src.selector import select_holdings


def test_broad_constraint_reported():
    pool = pd.DataFrame({'index_code': ['000300', '399006', 'X1', 'X2', 'X3'],
                         'flow_pct': [0.1, 0.2, 0.3, 0.4, 0.5]})
    name_map = {'000300': '沪深300', '399006': '创业板', 'X1': '医药', 'X2': '科技', 'X3': '军工'}
    holdings, diag = select_holdings(pool, StrategyConfig(), name_map)
    assert 'broad_ok' in diag
    assert diag['broad_n'] >= 1