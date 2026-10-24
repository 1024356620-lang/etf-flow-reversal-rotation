# -*- coding: utf-8 -*-
"""全局参数与常量。所有模块只从这里 import，不自行定义默认值。"""
from dataclasses import dataclass
from typing import FrozenSet, Dict, List
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'raw_data')

BROAD_BASE_CODES: FrozenSet[str] = frozenset({
    '000016', '000300', '000688', '000852', '000905', '000985',
    '399001', '399005', '399006', '399673', '399989', '399975',
    '399300', '399905',
})

BROAD_DEDUP: Dict[str, str] = {'399300': '000300', '399905': '000905'}

SECTOR_KEYWORDS: Dict[str, List[str]] = {
    '宽基': ['沪深300', '中证500', '中证1000', '中证2000', '中证800', '中证全指',
             '创业板', '科创', '上证50', '上证180', 'A100', 'A50', '国证2000',
             '深证100', '深证成指', '中小100', '1000', '500', '300'],
    '金融地产': ['金融', '银行', '证券', '保险', '地产', '房地产'],
    '医药': ['医药', '医疗', '生物', '创新药', '疫苗'],
    '科技': ['半导体', '芯片', '科技', '算力', '人工智能', 'AI', '软件', '通信', '5G', '信息技术'],
    '新能源': ['新能源', '光伏', '锂电', '电池', '碳中和', '绿色电力'],
    '军工': ['军工', '国防', '航天'],
    '消费': ['消费', '食品', '饮料', '白酒', '家电', '农业', '粮食'],
    '周期': ['有色', '煤炭', '钢铁', '化工', '能源', '石油', '资源'],
    '红利': ['红利', '价值'],
    '成长': ['成长', '创科技', '创业大盘'],
}


@dataclass(frozen=True)
class StrategyConfig:
    """策略配置。flow_filter_mode 用于区分复现版、实验 B、实验 C。"""
    start_date: str = '2019-01-01'
    end_date: str = '2026-01-31'
    rebalance_freq: str = 'M'

    top_n: int = 5
    factor_window: int = 10
    factor_min_periods: int = 5

    min_etf_size: float = 4e8
    max_inst_ratio: float = 0.70
    inst_lag_months: int = 2

    cost_rate: float = 0.001
    min_broad_n: int = 3
    max_per_sector: int = 99

    # 'absolute'       = flow_cum <= 0（复现版 A）
    # 'relative'       = 不设阈值，按 flow_pct 排序（实验 B）
    # 'percentile'     = flow_pct <= 0.5（旧改进版）
    # 'c_equal_weight' = 用 A 的总仓位，等权持有全部合格候选（实验 C）
    flow_filter_mode: str = 'absolute'

    # 't0' = 用 T 日数据，T 日收盘调仓（当前基线，可能含前视）
    # 't1' = 用截至 T-1 日的数据，T 日收盘调仓（保守版）
    signal_lag: str = 't0'

    risk_free_rate: float = 0.0