"""Replay ETF strategy A with a CSI 300 spot-return hedge proxy."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_inputs(directory, returns_path):
    account = pd.read_csv(directory / 'account.csv', parse_dates=['trade_date']).set_index('trade_date')
    holdings = pd.read_csv(directory / 'holdings.csv', parse_dates=['trade_date'],
                           dtype={'holdings': str}).set_index('trade_date')
    if (not account.index.is_unique or not account.index.is_monotonic_increasing
            or not holdings.index.is_unique or not holdings.index.isin(account.index).all()):
        raise ValueError('Invalid accounting or rebalance calendar')
    picks = {date: value.split(',') if pd.notna(value) else []
             for date, value in holdings.holdings.items()}
    codes = sorted({code for names in picks.values() for code in names} | {'000300'})
    chunks = pd.read_csv(returns_path, dtype={'index_code': str}, parse_dates=['trade_date'],
                         usecols=['trade_date', 'index_code', 'index_ret'], chunksize=200000)
    data = pd.concat(chunk.loc[chunk.index_code.isin(codes)] for chunk in chunks)
    prices = data.pivot(index='trade_date', columns='index_code', values='index_ret')
    prices = prices.reindex(index=account.index, columns=codes)
    # The first close has no prior return and the account starts in cash.
    market = prices['000300'].copy()
    if pd.isna(market.iloc[0]):
        market.iloc[0] = 0
    if not np.isfinite(market).all():
        raise ValueError('Incomplete CSI 300 history')
    audit_path = directory / 'actual_returns.csv'
    if audit_path.exists():
        audit = pd.read_csv(audit_path, parse_dates=['trade_date'], dtype={'index_code': str})
        actual = audit.pivot(index='trade_date', columns='index_code', values='actual_ret')
        prices.update(actual)
    schedule = {}
    for date, names in picks.items():
        if len(names) > 5 or len(names) != len(set(names)):
            raise ValueError(f'Invalid A holdings on {date}')
        schedule[account.index.get_loc(date)] = prices.columns.get_indexer(names)
    return account, prices.to_numpy(), market, schedule


def replay(returns, market, schedule, long_scale, hedge_ratio=0., annual_drag=0.):
    weights = np.zeros(returns.shape[1])
    short = 0.
    rows = []
    for day, (asset_returns, market_return) in enumerate(zip(returns, market)):
        if not np.isfinite(asset_returns[weights != 0]).all():
            raise ValueError(f'Missing invested return on {market.index[day]}')
        asset_returns = np.where(weights != 0, asset_returns, 0.)
        growth = 1 + weights @ asset_returns - short * (market_return + annual_drag / 252)
        weights = weights * (1 + asset_returns) / growth
        short = short * (1 + market_return) / growth
        fee = 0.
        if day in schedule:
            target = np.zeros_like(weights)
            target[schedule[day]] = long_scale / 5
            new_short = target.sum() * hedge_ratio
            # Each monthly roll closes the old short and opens the new short.
            fee = .001 * np.abs(target - weights).sum() + .0002 * (short + new_short)
            weights, short = target, new_short
        rows.append((growth * (1 - fee) - 1, weights.sum(), short, growth * fee))
    return pd.DataFrame(rows, index=market.index,
                        columns=['net_ret', 'long_weight', 'short_notional', 'cost'])


def metrics(daily, market):
    ret = daily.net_ret
    nav = (1 + ret).cumprod()
    vol = ret.std() * np.sqrt(252)
    return dict(annual_return=nav.iloc[-1] ** (252 / len(nav)) - 1,
                max_drawdown=(nav / nav.cummax().clip(lower=1) - 1).min(),
                annual_vol=vol, sharpe=ret.mean() * 252 / vol if vol else np.nan,
                beta=ret.cov(market) / market.var())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results_A'))
    parser.add_argument('--index-returns', type=Path, help='Full CSV or filtered CSV/ZIP')
    parser.add_argument('--out-dir', type=Path, default=Path('results_hedge'))
    args = parser.parse_args()
    account, returns, market, schedule = read_inputs(
        args.results_dir, args.index_returns or args.results_dir / 'index_returns.csv')
    def run(scale, hedge=0., drag=0.):
        return replay(returns, market, schedule, scale, hedge, drag)

    baseline = run(1.)
    np.testing.assert_allclose(baseline.net_ret, account.net_ret, atol=1e-10, rtol=0,
                               err_msg='A daily ledger reconciliation failed')
    hedged = run(.8, .5)
    target = metrics(hedged, market)
    scenarios = {'A': baseline, 'A_80pct': run(.8), 'hedge_proxy': hedged,
                 'hedge_drag_2pct': run(.8, .5, .02), 'hedge_drag_4pct': run(.8, .5, .04)}
    for metric in ['annual_vol', 'beta']:
        lo, hi = 0., 1.
        if not 0 <= target[metric] <= metrics(baseline, market)[metric]:
            raise ValueError(f'No fixed-exposure match for {metric}')
        for _ in range(24):
            mid = (lo + hi) / 2
            if metrics(run(mid), market)[metric] < target[metric]:
                lo = mid
            else:
                hi = mid
        scenarios['matched_' + metric] = run((lo + hi) / 2)
    summary = pd.DataFrame({name: metrics(frame, market)
                            for name, frame in scenarios.items()}).T
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / 'hedge_summary.csv', index_label='scenario')
    pd.concat(scenarios, names=['scenario']).to_csv(args.out_dir / 'hedge_daily.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
