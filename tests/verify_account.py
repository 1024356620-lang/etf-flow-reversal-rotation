# -*- coding: utf-8 -*-
"""核验 account / daily_weights / cash_weights / rebalance_detail / holdings_returns。

口径：
  - 权重占总资产，sum(w) + cash = 1
  - 漂移 w_t = w_{t-1}(1+r_t)/(1+R_t)，cash 同理
  - market_ret 复算：(w_{t-1} * r_t).sum()
  - turnover 含现金；traded_fraction 只含证券
  - fee_fraction = traded_fraction * cost_rate（占当日扣费前净资产）
  - cost = (1 + market_ret) * fee_fraction（占上日净资产）
  - net_ret = market_ret - cost
  - 选仓约束：n_selected<=5, non_broad_n<=2, 宽基不足时 n_selected<=broad_n+2

失败返回非零退出码。
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'
COST_RATE = 0.001
TOL = 1e-10


def _parse_weights(s: str) -> dict:
    if not isinstance(s, str) or not s.strip():
        return {}
    out = {}
    for item in s.split(','):
        code, w = item.split(':')
        out[code.strip()] = float(w)
    return out


def main() -> bool:
    failures = []

    acct = pd.read_csv(RESULTS / 'account.csv', parse_dates=['trade_date']).set_index('trade_date')
    dw = pd.read_csv(RESULTS / 'daily_weights.csv', parse_dates=['trade_date'])
    cw = pd.read_csv(RESULTS / 'cash_weights.csv', parse_dates=['trade_date']).set_index('trade_date')
    rb = pd.read_csv(RESULTS / 'rebalance_detail.csv', parse_dates=['trade_date'])
    hr = pd.read_csv(RESULTS / 'holdings_returns.csv', parse_dates=['trade_date'])
    diag = pd.read_csv(RESULTS / 'diagnostics.csv', parse_dates=['trade_date'])

    ret_panel = hr.pivot(index='trade_date', columns='index_code', values='index_ret')
    w_pivot = (dw.pivot(index='trade_date', columns='index_code', values='weight')
                 .fillna(0.0))

    rebalance_dates = set(rb['trade_date'])
    all_dates = sorted(acct.index)

    def fail(msg):
        failures.append(msg)
        print(f'  FAIL: {msg}')

    # [0] 资产 + 现金 = 1
    print('\n[0] 每日资产权重 + 现金权重 = 1')
    total = w_pivot.sum(axis=1).reindex(cw.index).fillna(0.0) + cw['cash_weight']
    diff_total = (total - 1.0).abs()
    if diff_total.max() > TOL:
        fail(f'权重和偏离 1: 最大 {diff_total.max():.3e}')
    else:
        print(f'  全部 {len(total)} 天满足，最大偏差 {diff_total.max():.3e}')

    # [A] 收益覆盖
    print('\n[A] 收益覆盖')
    missing_ret, prev_w = [], pd.Series(dtype=float)
    for d in all_dates:
        if len(prev_w) > 0:
            r_today = ret_panel.loc[d] if d in ret_panel.index else pd.Series(dtype=float)
            for code, w in prev_w.items():
                if w > 0 and (code not in r_today.index or
                              not np.isfinite(r_today.get(code, np.nan))):
                    missing_ret.append((d, code))
        day_w = w_pivot.loc[d] if d in w_pivot.index else pd.Series(dtype=float)
        prev_w = day_w[day_w > 0]
    dropped = sorted(set(w_pivot.columns[w_pivot.sum(axis=0) > 0]) - set(ret_panel.columns))
    if dropped:
        fail(f'权重中有 {len(dropped)} 个代码在收益表中不存在: {dropped[:10]}')
    if missing_ret:
        fail(f'收益缺失/非有限: {len(missing_ret)} 条，示例 {missing_ret[:5]}')
    else:
        print('  上日权重 > 0 的资产收益全部存在且有限。')
    first_rb = rb['trade_date'].min()
    missing_dates = [d for d in all_dates
                     if d not in set(hr['trade_date'].unique()) and d > first_rb]
    if missing_dates:
        fail(f'holdings_returns 未覆盖 {len(missing_dates)} 天')
    else:
        print('  holdings_returns 覆盖账本全部持仓日期。')

    # [B] market_ret 复算
    print('\n[B] market_ret 复算')
    recon = (w_pivot.shift(1) * ret_panel.reindex(w_pivot.index)).sum(axis=1)
    common = recon.dropna().index.intersection(acct.index)
    diff_mr = (acct.loc[common, 'market_ret'] - recon.loc[common]).abs()
    print(f'  样本 {len(common)} 天，最大误差 {diff_mr.max():.3e}')
    if diff_mr.max() > TOL:
        fail(f'market_ret 最大误差 {diff_mr.max():.3e}')

    # [C] 权重漂移
    print('\n[C] 权重漂移')
    rb_by_date = {r['trade_date']: r for _, r in rb.iterrows()}
    drift_non, drift_old, post = [], [], []
    prev_w, prev_cash = pd.Series(dtype=float), 1.0
    for d in all_dates:
        r_today = ret_panel.loc[d] if d in ret_panel.index else pd.Series(dtype=float)
        day_w = w_pivot.loc[d] if d in w_pivot.index else pd.Series(dtype=float)
        day_cash = float(cw.loc[d, 'cash_weight']) if d in cw.index else 1.0
        if len(prev_w) > 0 or prev_cash > 0:
            R = float((prev_w * r_today.reindex(prev_w.index).fillna(0.0)).sum())
            denom = 1.0 + R
            exp_w = prev_w * (1 + r_today.reindex(prev_w.index).fillna(0.0)) / denom
            exp_cash = prev_cash / denom
            if d in rebalance_dates:
                old_w = pd.Series(_parse_weights(rb_by_date[d]['old_weights']))
                codes = set(old_w.index) | set(exp_w.index)
                diff = (old_w.reindex(codes).fillna(0) - exp_w.reindex(codes).fillna(0)).abs().max()
                if diff > TOL:
                    drift_old.append((d, float(diff)))
                if abs(float(rb_by_date[d]['old_cash_weight']) - exp_cash) > TOL:
                    drift_old.append((d, abs(float(rb_by_date[d]['old_cash_weight']) - exp_cash)))
                new_w = pd.Series(_parse_weights(rb_by_date[d]['new_weights']))
                codes = set(day_w.index) | set(new_w.index)
                diff_post = (day_w.reindex(codes).fillna(0) - new_w.reindex(codes).fillna(0)).abs().max()
                if diff_post > TOL:
                    post.append((d, float(diff_post)))
                if abs(day_cash - float(rb_by_date[d]['new_cash_weight'])) > TOL:
                    post.append((d, abs(day_cash - float(rb_by_date[d]['new_cash_weight']))))
            else:
                codes = set(day_w.index) | set(exp_w.index)
                diff = (day_w.reindex(codes).fillna(0) - exp_w.reindex(codes).fillna(0)).abs().max()
                if diff > TOL:
                    drift_non.append((d, float(diff)))
                if abs(day_cash - exp_cash) > TOL:
                    drift_non.append((d, abs(day_cash - exp_cash)))
        prev_w = day_w[day_w > 0]
        prev_cash = day_cash
    if drift_non:
        fail(f'非调仓日漂移不符: {len(drift_non)} 天')
    else:
        print('  非调仓日漂移全部满足。')
    if drift_old:
        fail(f'调仓日 old 不符: {len(drift_old)} 天')
    else:
        print('  调仓日 old_weights/old_cash 一致。')
    if post:
        fail(f'调仓后 new 不符: {len(post)} 天')
    else:
        print('  调仓后 daily_weights/cash 与 new 一致。')

    # [D] 换手 / 成本
    print('\n[D] 换手 / 成本复算')
    t_fail, c_fail = [], []
    for _, r in rb.iterrows():
        old_w = pd.Series(_parse_weights(r['old_weights']))
        new_w = pd.Series(_parse_weights(r['new_weights']))
        old_full = old_w.copy(); old_full['__cash__'] = float(r['old_cash_weight'])
        new_full = new_w.copy(); new_full['__cash__'] = float(r['new_cash_weight'])
        codes = set(old_full.index) | set(new_full.index)
        turnover = (new_full.reindex(codes).fillna(0) -
                    old_full.reindex(codes).fillna(0)).abs().sum() / 2.0
        sec = set(old_w.index) | set(new_w.index)
        traded = (new_w.reindex(sec).fillna(0) - old_w.reindex(sec).fillna(0)).abs().sum()
        fee = traded * COST_RATE
        mr = float(acct.loc[r['trade_date'], 'market_ret'])
        cost = (1.0 + mr) * fee
        if abs(turnover - float(r['turnover'])) > TOL:
            t_fail.append((r['trade_date'], turnover, float(r['turnover'])))
        if abs(cost - float(r['cost'])) > TOL:
            c_fail.append((r['trade_date'], cost, float(r['cost'])))
    if t_fail:
        fail(f'turnover 复算不符: {len(t_fail)} 次，示例 {t_fail[:3]}')
    else:
        print('  turnover 复算一致（含现金）。')
    if c_fail:
        fail(f'cost 复算不符: {len(c_fail)} 次，示例 {c_fail[:3]}')
    else:
        print('  cost 复算一致（只含证券，基数换算到上日净资产）。')
    merged = acct.join(rb.set_index('trade_date')[['turnover', 'cost']],
                       how='inner', rsuffix='_rb')
    if (merged['turnover'] - merged['turnover_rb']).abs().max() > TOL:
        fail('account vs rebalance turnover 不一致')
    if (merged['cost'] - merged['cost_rb']).abs().max() > TOL:
        fail('account vs rebalance cost 不一致')
    else:
        print('  account 与 rebalance_detail 的 turnover/cost 一致。')

    # [D2] net_ret 与成本基数
    print('\n[D2] net_ret = market_ret - cost，cost = (1+market_ret)*fee_fraction')
    diff_net = (acct['net_ret'] - (acct['market_ret'] - acct['cost'])).abs()
    if diff_net.max() > TOL:
        fail(f'net_ret != market_ret - cost: 最大 {diff_net.max():.3e}')
    else:
        print(f'  net_ret = market_ret - cost 全部满足，最大偏差 {diff_net.max():.3e}')

    c_base_fail = []
    for _, r in rb.iterrows():
        mr = float(acct.loc[r['trade_date'], 'market_ret'])
        expected = (1.0 + mr) * float(r['traded_fraction']) * COST_RATE
        if abs(expected - float(r['cost'])) > TOL:
            c_base_fail.append((r['trade_date'], expected, float(r['cost'])))
    if c_base_fail:
        fail(f'cost 基数换算不符: {len(c_base_fail)} 次，示例 {c_base_fail[:3]}')
    else:
        print('  cost = (1+market_ret)*traded_fraction*cost_rate 全部一致。')

    # [E] 选仓约束
    print('\n[E] 选仓约束检查')
    bad = diag[(diag['n_selected'] > 5) |
               (diag.get('non_broad_n', pd.Series(0, index=diag.index)) > 2) |
               ((diag['broad_n'] < 3) & (diag['n_selected'] > diag['broad_n'] + 2))]
    if not bad.empty:
        fail(f'选仓约束失效: {len(bad)} 次，示例 '
             f'{bad[["trade_date","broad_n","n_selected"]].head().to_dict("records")}')
    else:
        print('  n_selected<=5, non_broad_n<=2, 宽基不足时 n_selected<=broad_n+2 全部满足。')

    print('\n结论:', 'PASS' if not failures else f'FAIL（{len(failures)} 项）')
    return not failures


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)