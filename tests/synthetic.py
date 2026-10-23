# tests/synthetic.py
import numpy as np
import pandas as pd

from src.data_loader import MarketData


def make_returns(dates, codes, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(codes))),
                        index=dates, columns=codes)


def simple_calendar(n=60, start='2024-01-01'):
    return pd.bdate_range(start, periods=n)


def make_market_data(n_days=250, n_codes=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_days)
    codes = [f'{i:06d}' for i in range(1, n_codes + 1)]

    # 指数日行情 + 完整收益面板
    rets = rng.normal(0, 0.01, (n_days, n_codes))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    index_daily = pd.DataFrame({
        'index_code': np.repeat(codes, n_days),
        'trade_date': np.tile(dates, n_codes),
        'close': prices.T.reshape(-1),
    })
    index_daily['index_ret'] = index_daily.groupby('index_code')['close'].pct_change()
    full_returns = index_daily.pivot_table(
        index='trade_date', columns='index_code', values='index_ret').reindex(dates)

    # ETF 日行情：每个指数配一个 ETF
    etf_codes = [f'5{i:05d}' for i in range(1, n_codes + 1)]
    mapping = pd.DataFrame({'etf_code': etf_codes, 'index_code': codes})
    shares = np.full((n_days, n_codes), 1e8)
    # 随机申赎，制造资金流
    shares += np.cumsum(rng.normal(0, 1e6, (n_days, n_codes)), axis=0)
    etf_daily = pd.DataFrame({
        'etf_code': np.repeat(etf_codes, n_days),
        'trade_date': np.tile(dates, n_codes),
        'close': prices.T.reshape(-1),
        'shares': shares.T.reshape(-1),
    })
    etf_daily['share_chg'] = etf_daily.groupby('etf_code')['shares'].diff()
    etf_daily['etf_flow'] = etf_daily['share_chg'] * etf_daily['close']
    etf_daily['etf_size'] = etf_daily['shares'] * etf_daily['close']

    # 机构占比：每季度一次，全部低于阈值
    report_dates = pd.date_range('2022-12-31', periods=8, freq='QE')
    inst_ratio = pd.DataFrame({
        'etf_code': np.repeat(etf_codes, len(report_dates)),
        'report_date': np.tile(report_dates, n_codes),
        'inst_ratio': rng.uniform(0.1, 0.6, n_codes * len(report_dates)),
    })
    inst_ratio['available_date'] = inst_ratio['report_date'] + pd.DateOffset(months=2)

    index_name = pd.DataFrame({
        'index_code': codes,
        'index_name': ['沪深300', '创业板', '医药', '科技', '军工'][:n_codes],
    })

    benchmark = pd.DataFrame({'trade_date': dates,
                              'index_ret': rets.mean(axis=1)})

    return MarketData(etf_daily, index_daily, inst_ratio, mapping, index_name,
                      benchmark, dates, full_returns)