# -*- coding: utf-8 -*-
"""绩效指标。"""
import numpy as np
import pandas as pd


def evaluate(net_returns: pd.Series,
             benchmark_returns: pd.Series = None,
             risk_free_rate: float = 0.0,
             freq: int = 252) -> dict:
    ret = net_returns.dropna()
    if len(ret) == 0:
        return {}
    nav = pd.concat([pd.Series([1.0]), (1 + ret).cumprod().reset_index(drop=True)],
                    ignore_index=True)
    ann_ret = nav.iloc[-1] ** (freq / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(freq)

    # 标准夏普：日度超额均值 / 日度标准差 × √252
    rf_daily = (1 + risk_free_rate) ** (1 / freq) - 1
    excess = ret - rf_daily
    sharpe = excess.mean() / ret.std() * np.sqrt(freq) if ret.std() > 0 else np.nan

    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    result = {
        '年化收益': f'{ann_ret:.2%}', '年化波动': f'{ann_vol:.2%}',
        '夏普': f'{sharpe:.2f}', '最大回撤': f'{max_dd:.2%}',
        'Calmar': f'{calmar:.2f}', '日度胜率': f'{(ret > 0).mean():.2%}',
    }

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(ret.index)   # 不对齐则 NaN，不 ffill
        common = ret.index.intersection(bench.dropna().index)
        if len(common) > 0:
            active = ret.loc[common] - bench.loc[common]
            ann_active = (1 + active).prod() ** (freq / len(active)) - 1
            result['年化主动收益'] = f'{ann_active:.2%}'
            result['主动收益口径'] = '日度主动收益几何年化（非净值差）'
    return result


def yearly_performance(net_returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'ret': net_returns.dropna()})
    df['year'] = df.index.year
    rows = []
    for year, g in df.groupby('year'):
        if len(g) < 2:
            continue
        nav = pd.concat([pd.Series([1.0]), (1 + g['ret']).cumprod().reset_index(drop=True)],
                        ignore_index=True)
        ann_ret = nav.iloc[-1] - 1
        dd = (nav / nav.cummax() - 1).min()
        vol = g['ret'].std() * np.sqrt(252)
        sharpe = (g['ret'].mean() * 252) / vol if vol > 0 else np.nan
        rows.append({'年份': year, '收益': f'{ann_ret:.2%}',
                     '最大回撤': f'{dd:.2%}', '夏普': f'{sharpe:.2f}'})
    return pd.DataFrame(rows)