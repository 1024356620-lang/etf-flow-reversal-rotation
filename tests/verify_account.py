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

跨空仓期处理：
  - w_pivot 只含有持仓的日期，直接 shift(1) 会跨过空仓期，错位
  - 本脚本先把 w_pivot 和 cw 补齐到完整账本日期（acct.index），
    空仓日资产权重为 0、现金权重为 1，再 shift(1) 得到上一交易日权重
  - [A]、[B]、[C] 全部使用补齐后的 prev_weights / prev_cash

失败返回非零退出码。
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'
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

    all_dates = list(acct.index)
    rebalance_dates = set(rb['trade_date'])

    # ---------- 关键：补齐到完整账本日期，再 shift ----------
    # 空仓日资产权重为 0，现金权重为 1
    w_daily = w_pivot.reindex(all_dates, fill_value=0.0)
    cw_daily = cw['cash_weight'].reindex(all_dates, fill_value=1.0)
    prev_weights = w_daily.shift(1, fill_value=0.0)   # 上一交易日资产权重
    prev_cash = cw_daily.shift(1, fill_value=1.0)     # 上一交易日现金权重

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

    # ================= [A] 收益覆盖 =================
    print('\n[A] 收益覆盖（按完整账本日期，上日权重 > 0 的资产当日收益必须存在且有限）')
    missing_ret = []
    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pw = pw[pw > 0]
        if len(pw) == 0:
            continue
        r_today = ret_panel.loc[d] if d in ret_panel.index else pd.Series(dtype=float)
        for code, w in pw.items():
            if code not in r_today.index or not np.isfinite(r_today.get(code, np.nan)):
                missing_ret.append((d, code))

    w_codes = set(w_daily.columns[w_daily.sum(axis=0) > 0])
    r_codes = set(ret_panel.columns)
    dropped = sorted(w_codes - r_codes)
    if dropped:
        fail(f'权重中有 {len(dropped)} 个代码在收益表中不存在: {dropped[:10]}')
    if missing_ret:
        fail(f'上日权重 > 0 但当日收益缺失/非有限: {len(missing_ret)} 条，示例 {missing_ret[:5]}')
    else:
        print('  上日权重 > 0 的资产收益全部存在且有限。')

    # holdings_returns 覆盖：上日有持仓的日期必须出现在 hr 中
    hr_dates = set(hr['trade_date'].unique())
    missing_dates = []
    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pw = pw[pw > 0]
        if len(pw) == 0:
            continue   # 上日空仓，不需要 holdings_returns
        if d not in hr_dates:
            missing_dates.append(d)
    if missing_dates:
        fail(f'holdings_returns 未覆盖 {len(missing_dates)} 个“上日有持仓”的日期，'
             f'示例 {missing_dates[:5]}')
    else:
        print('  holdings_returns 覆盖全部“上日有持仓”的日期。')

    # ================= [B] market_ret 复算 =================
    print('\n[B] market_ret 复算（补齐后前一日权重 × 当日收益）')
    recon = (prev_weights * ret_panel.reindex(all_dates)).sum(axis=1)
    diff_mr = (acct['market_ret'] - recon).abs()
    print(f'  样本 {len(diff_mr)} 天（完整账本），最大误差 {diff_mr.max():.3e}')
    if diff_mr.max() > TOL:
        fail(f'market_ret 最大误差 {diff_mr.max():.3e}')
        bad = diff_mr[diff_mr > TOL].head(5)
        print(f'  误差 > TOL 的日期示例: {bad.to_dict()}')

    # ================= [C] 权重漂移（含空仓期） =================
    print('\n[C] 权重漂移（w_t = w_{t-1}(1+r_t)/(1+R_t)，cash 同理）')
    rb_by_date = {r['trade_date']: r for _, r in rb.iterrows()}
    drift_non, drift_old, post = [], [], []

    for d in all_dates[1:]:
        pw = prev_weights.loc[d]
        pc = float(prev_cash.loc[d])
        # 上一交易日无持仓且满现金，则当日若仍空仓，漂移是恒等；若当日调仓，old 应为空
        r_today = ret_panel.loc[d] if d in ret_panel.index else pd.Series(dtype=float)
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

    # ================= [D] 换手 / 成本（含空仓分支） =================
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