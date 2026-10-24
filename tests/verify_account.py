# -*- coding: utf-8 -*-
"""核验账本内部一致性、信号缺失处理与漂移复算。

核验范围：
  - [0] 资产权重 + 现金权重 = 1
  - [A] 原始数据缺失覆盖检查（使用 holdings_returns.csv）
  - [A2] 实际记账收益完整性（使用 actual_returns.csv）
  - [B] market_ret 复算（使用实际记账收益）
  - [C] 权重漂移复算（使用实际记账收益）
  - [D] 换手与成本复算
  - [D2] net_ret 恒等式
  - [E] 选仓约束检查

用法：
  python tests/verify_account.py --results-dir results_A
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COST_RATE = 0.001
TOL = 1e-10


def _parse_weights(s) -> dict:
    if not isinstance(s, str) or not s.strip():
        return {}
    out = {}
    for item in s.split(','):
        code, w = item.split(':')
        out[code.strip()] = float(w)
    return out


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', help='账本所在目录')
    args = parser.parse_args()

    results = Path(args.results_dir)
    if not results.exists():
        print(f'目录不存在: {results}')
        return False

    acct = pd.read_csv(results / 'account.csv', parse_dates=['trade_date']).set_index('trade_date')
    dw = pd.read_csv(results / 'daily_weights.csv', parse_dates=['trade_date'])
    cw = pd.read_csv(results / 'cash_weights.csv', parse_dates=['trade_date']).set_index('trade_date')
    rb = pd.read_csv(results / 'rebalance_detail.csv', parse_dates=['trade_date'])
    hr = pd.read_csv(results / 'holdings_returns.csv', parse_dates=['trade_date'])
    
    ar_path = results / 'actual_returns.csv'
    if not ar_path.exists():
        print(f'缺少 {ar_path}，无法核验实际记账收益')
        return False
    ar = pd.read_csv(ar_path, parse_dates=['trade_date'])

    diag = pd.read_csv(results / 'diagnostics.csv', parse_dates=['trade_date'])

    # 收益面板
    ret_raw = hr.pivot(index='trade_date', columns='index_code', values='index_ret')
    ret_actual = ar.pivot(index='trade_date', columns='index_code', values='actual_ret')
    
    w_pivot = (dw.pivot(index='trade_date', columns='index_code', values='weight')
                 .fillna(0.0))

    all_dates = list(acct.index)
    rebalance_dates = set(rb['trade_date'])

    # 补齐到完整账本日期
    w_daily = w_pivot.reindex(all_dates, fill_value=0.0)
    cw_daily = cw['cash_weight'].reindex(all_dates, fill_value=1.0)
    prev_weights = w_daily.shift(1, fill_value=0.0)
    prev_cash = cw_daily.shift(1, fill_value=1.0)

    failures = []

    def fail(msg):
        failures.append(msg)
        print(f'  FAIL: {msg}')

    # ================= [0] 资产 + 现金 = 1 =================
    print('\n[0] 每日资产权重 + 现金权重 = 1')
    total = w_daily.sum(axis=1) + cw_daily
    diff_total = (total - 1.0).abs()
    if diff_total.max() > TOL:
        fail(f'权重和偏离 1: 最大 {diff_total.max():.3e}')
    else:
        print(f'  全部 {len(total)} 天满足，最大偏差 {diff_total.max():.3e}')

    # ================= [A] 原始收益覆盖 =================
    print('\n[A] 原始收益覆盖（上日权重 > 0 的资产，当日原始收益缺失检查）')
    missing_ret = []
    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pw = pw[pw > 0]
        if len(pw) == 0:
            continue
        r_today = ret_raw.loc[d] if d in ret_raw.index else pd.Series(dtype=float)
        for code, w in pw.items():
            if code not in r_today.index or not np.isfinite(r_today.get(code, np.nan)):
                missing_ret.append((d, code))

    w_codes = set(w_daily.columns[w_daily.sum(axis=0) > 0])
    r_codes = set(ret_raw.columns)
    dropped = sorted(w_codes - r_codes)
    if dropped:
        fail(f'权重中有 {len(dropped)} 个代码在原始收益表中不存在: {dropped[:10]}')
    if missing_ret:
        print(f'  注意: 发现 {len(missing_ret)} 条原始收益缺失，应由回测引擎填补。')
        # 验证这些缺失是否确实存在于实际的 missing_returns.csv 里（可选，视需求）
    else:
        print('  上日权重 > 0 的资产原始收益全部存在且有限。')

    # ================= [A2] 实际记账收益完整性 =================
    print('\n[A2] 实际记账收益完整性')
    missing_actual = []

    for d in all_dates[1:]:
        # 当天是否持仓不影响检查：清仓日也要计算旧持仓收益
        pw = prev_weights.loc[d]
        held_codes = pw.index[pw > 0]

        if len(held_codes) == 0:
            continue

        r_today = (
            ret_actual.loc[d]
            if d in ret_actual.index
            else pd.Series(dtype=float)
        )

        values = r_today.reindex(held_codes)
        bad = values[~np.isfinite(values)]

        missing_actual.extend((d, code) for code in bad.index)

    if missing_actual:
        fail(
            f'实际记账收益缺失或非有限，共 {len(missing_actual)} 条，'
            f'示例：{missing_actual[:10]}'
        )
        # 禁止继续用 fillna(0) 掩盖缺失
        print('\n结论: FAIL')
        return False

    print('  所有上一日持仓资产的当日实际收益均存在且有限，包含清仓日。')

    # ================= [B] market_ret 复算 =================
    print('\n[B] market_ret 复算（使用实际记账收益）')
    # 用实际的 ret_actual 来复算 market_ret
    recon = (prev_weights * ret_actual.reindex(all_dates)).sum(axis=1)
    diff_mr = (acct['market_ret'] - recon).abs()
    print(f'  样本 {len(diff_mr)} 天，最大误差 {diff_mr.max():.3e}')
    if diff_mr.max() > TOL:
        bad = diff_mr[diff_mr > TOL].head(5)
        fail(f'market_ret 最大误差 {diff_mr.max():.3e}，示例日期: {bad.index.tolist()}')

    # ================= [C] 权重漂移（含空仓期） =================
    print('\n[C] 权重漂移（w_t = w_{t-1}(1+r_t)/(1+R_t)，使用实际记账收益）')
    rb_by_date = {r['trade_date']: r for _, r in rb.iterrows()}
    drift_non, drift_old, post = [], [], []

    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pc = float(prev_cash.loc[d])
        
        r_today = ret_actual.loc[d] if d in ret_actual.index else pd.Series(dtype=float)
        R = float((pw * r_today.reindex(pw.index).fillna(0.0)).sum())
        denom = 1.0 + R
        exp_w = pw * (1 + r_today.reindex(pw.index).fillna(0.0)) / denom if denom != 0 else pw
        exp_cash = pc / denom if denom != 0 else pc

        day_w = w_daily.loc[d]
        day_cash = float(cw_daily.loc[d])

        if d in rebalance_dates:
            old_w = pd.Series(_parse_weights(rb_by_date[d]['old_weights']))
            codes = set(old_w.index) | set(exp_w.index)
            diff = (old_w.reindex(codes).fillna(0) - exp_w.reindex(codes).fillna(0)).abs().max()
            if diff > TOL:
                drift_old.append((d, float(diff)))
            old_cash_file = float(rb_by_date[d]['old_cash_weight'])
            if abs(old_cash_file - exp_cash) > TOL:
                drift_old.append((d, abs(old_cash_file - exp_cash)))

            new_w = pd.Series(_parse_weights(rb_by_date[d]['new_weights']))
            codes = set(day_w.index) | set(new_w.index)
            diff_post = (day_w.reindex(codes).fillna(0) - new_w.reindex(codes).fillna(0)).abs().max()
            if diff_post > TOL:
                post.append((d, float(diff_post)))
            new_cash_file = float(rb_by_date[d]['new_cash_weight'])
            if abs(day_cash - new_cash_file) > TOL:
                post.append((d, abs(day_cash - new_cash_file)))
        else:
            codes = set(day_w.index) | set(exp_w.index)
            diff = (day_w.reindex(codes).fillna(0) - exp_w.reindex(codes).fillna(0)).abs().max()
            if diff > TOL:
                drift_non.append((d, float(diff)))
            if abs(day_cash - exp_cash) > TOL:
                drift_non.append((d, abs(day_cash - exp_cash)))

    if drift_non:
        fail(f'非调仓日漂移不符: {len(drift_non)} 天，示例 {drift_non[:3]}')
    else:
        print('  非调仓日漂移全部满足（含空仓期恒等）。')
    if drift_old:
        fail(f'调仓日 old 不符: {len(drift_old)} 天，示例 {drift_old[:3]}')
    else:
        print('  调仓日 old_weights/old_cash 一致（含清仓、建仓边界）。')
    if post:
        fail(f'调仓后 new 不符: {len(post)} 天，示例 {post[:3]}')
    else:
        print('  调仓后 daily_weights/cash 与 new 一致。')

    # ================= [D] 换手 / 成本 =================
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

    # ================= [D2] net_ret 与成本基数 =================
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

    # ================= [E] 选仓约束 =================
    print('\n[E] 选仓约束检查')
    bad = diag[(diag['n_selected'] > 5) |
               (diag['non_broad_n'] > 2) |
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