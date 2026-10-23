# -*- coding: utf-8 -*-
"""选股规则。宽基约束：预留 min_broad_n 个宽基仓位，缺额保留现金，不以其他指数补足。

规则：
  1. 按资金流分位从低到高排序，优先满足 min_broad_n 个宽基。
  2. 宽基足够时补到 top_n；宽基不足时非宽基最多补 2 个。
  3. 缺额仓位由回测引擎按现金处理：gross = n_selected / top_n。
  4. 断言：n_selected <= top_n，non_broad_n <= 2，宽基不足时 n_selected <= broad_n + 2。
"""
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    StrategyConfig, BROAD_BASE_CODES, BROAD_DEDUP, SECTOR_KEYWORDS,
)

MAX_NON_BROAD_WHEN_SHORT = 2


def _sector_tag(name: str) -> str:
    if not isinstance(name, str):
        return '其他'
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(k in name for k in kws):
            return sector
    return '其他'


def _is_broad(code: str, name: str) -> bool:
    """宽基判断的唯一来源：代码集合 或 名称关键词，取并集。"""
    if code in BROAD_BASE_CODES:
        return True
    return _sector_tag(name) == '宽基'


def select_holdings(pool: pd.DataFrame,
                    config: StrategyConfig,
                    name_map: Dict[str, str],
                    sort_col: str = 'flow_cum_std') -> Tuple[List[str], Dict]:
    pool = pool.sort_values(sort_col)
    all_codes = pool['index_code'].tolist()
    is_broad = {c: _is_broad(c, name_map.get(c, '')) for c in all_codes}

    selected, sector_count, broad_selected = [], {}, set()
    broad_n = 0

    def try_add(code) -> bool:
        nonlocal broad_n
        if is_broad[code]:
            base = BROAD_DEDUP.get(code, code)
            if base in broad_selected:
                return False
        s = '宽基' if is_broad[code] else _sector_tag(name_map.get(code, ''))
        if sector_count.get(s, 0) >= config.max_per_sector:
            return False
        selected.append(code)
        sector_count[s] = sector_count.get(s, 0) + 1
        if is_broad[code]:
            broad_selected.add(BROAD_DEDUP.get(code, code))
            broad_n += 1
        return True

    # 1) 优先满足宽基
    for c in [x for x in all_codes if is_broad[x]]:
        if broad_n >= config.min_broad_n:
            break
        try_add(c)

    # 2) 补齐
    if broad_n >= config.min_broad_n:
        for c in all_codes:
            if len(selected) >= config.top_n:
                break
            if c not in selected:
                try_add(c)
        fallback = None
    else:
        non_broad_n = 0
        for c in all_codes:
            if c in selected or is_broad[c]:
                continue
            if non_broad_n >= MAX_NON_BROAD_WHEN_SHORT:
                break
            if try_add(c):
                non_broad_n += 1
        fallback = 'broad_shortfall_reduce_position'

    # 3) 断言
    non_broad_n = sum(1 for c in selected if not is_broad[c])
    assert len(selected) <= config.top_n, \
        f'n_selected={len(selected)} > top_n={config.top_n}'
    assert non_broad_n <= MAX_NON_BROAD_WHEN_SHORT, \
        f'non_broad_n={non_broad_n} > {MAX_NON_BROAD_WHEN_SHORT}'
    if broad_n < config.min_broad_n:
        assert len(selected) <= broad_n + MAX_NON_BROAD_WHEN_SHORT, \
            f'broad_n={broad_n}, n_selected={len(selected)} > broad_n+2'

    cash_slots = max(config.top_n - len(selected), 0)
    diagnostics = {
        'n_selected': len(selected),
        'broad_n': broad_n,
        'non_broad_n': non_broad_n,
        'broad_ok': broad_n >= config.min_broad_n,
        'fallback': fallback,
        'cash_slots': cash_slots,
        'cash_weight_implied': cash_slots / config.top_n,
        'broad_shortfall_reason': (
            None if broad_n >= config.min_broad_n
            else ('no_broad_candidates' if broad_n == 0 else 'insufficient_broad')
        ),
    }
    return selected, diagnostics