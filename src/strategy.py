# -*- coding: utf-8 -*-
"""
中泰证券 ETF资金流反转效应 指数轮动策略 复现
"""

import glob
import os
import re
import warnings
import pandas as pd
import numpy as np

# ==================== 参数 ====================
START_DATE = '2019-01-01'
END_DATE = '2026-01-31'
REBALANCE_FREQ = 'M'
TOP_N = 5
FACTOR_WINDOW = 10
FACTOR_MIN_PERIODS = 5
MIN_ETF_SIZE = 4e8
MAX_INST_RATIO = 0.70
COST_RATE = 0.001

MIN_BROAD_N = 3
MAX_PER_SECTOR = 99

BROAD_BASE_CODES = {
    '000016', '000300', '000688', '000852', '000905', '000985',
    '399001', '399005', '399006', '399673', '399989', '399975',
    '399300', '399905',
}

BROAD_DEDUP = {
    '399300': '000300',
    '399905': '000905',
}

SECTOR_KEYWORDS = {
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

# ==================== 数据目录设置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'raw_data')


# ==================== 工具 ====================
def _ensure_columns(df, rename_map):
    cols = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=cols)


def _clean_index_code(s):
    s = str(s).strip()
    if not s or s.lower() in ('nan', 'none', ''):
        return ''
    parts = s.split()
    if not parts:
        return ''
    s = parts[0]
    s = re.sub(r'\.(SZ|SH|CSI|WI)$', '', s, flags=re.IGNORECASE)
    return s


def _sector_tag(name):
    if not isinstance(name, str):
        return '其他'
    for sector, kws in SECTOR_KEYWORDS.items():
        if any(k in name for k in kws):
            return sector
    return '其他'


# ==================== 因子 ====================
def compute_index_flow(etf_daily, mapping):
    df = etf_daily.copy()
    df = _ensure_columns(df, {
        'Trdmnt': 'trade_date', 'TradingDate': 'trade_date', 'date': 'trade_date',
        'SecCode': 'etf_code', 'ETFCode': 'etf_code', 'code': 'etf_code', 'Symbol': 'etf_code',
        'Clsprc': 'close', 'ClosePrice': 'close', 'close_price': 'close', 'nav': 'close',
        'Shr': 'shares', 'Shares': 'shares', 'share': 'shares',
        'total_share': 'shares', 'CirculatedShare': 'shares',
    })
    mapping = _ensure_columns(mapping, {
        'SecCode': 'etf_code', 'ETFCode': 'etf_code', 'code': 'etf_code',
        'IndexCode': 'index_code', 'indexcode': 'index_code', 'idx_code': 'index_code',
    })

    if 'index_code' not in df.columns:
        df = df.merge(mapping[['etf_code', 'index_code']], on='etf_code', how='left')

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(['etf_code', 'trade_date'])

    if 'etf_flow' not in df.columns:
        df['share_chg'] = df.groupby('etf_code')['shares'].diff()
        df['etf_flow'] = df['share_chg'] * df['close']
    if 'etf_size' not in df.columns:
        df['etf_size'] = df['shares'] * df['close']

    df = df.dropna(subset=['index_code', 'close', 'etf_size'])
    df = df[df['etf_size'] > 0]

    agg = df.groupby(['index_code', 'trade_date']).agg(
        total_flow=('etf_flow', 'sum'),
        total_size=('etf_size', 'sum')
    ).reset_index()
    agg = agg.sort_values(['index_code', 'trade_date'])
    agg['prev_size'] = agg.groupby('index_code')['total_size'].shift(1)
    agg['flow_strength'] = agg['total_flow'] / agg['prev_size']
    agg['flow_strength'] = agg['flow_strength'].replace([np.inf, -np.inf], np.nan)
    return agg


def compute_flow_factor(index_flow, window=10, min_periods=5):
    df = index_flow.sort_values(['index_code', 'trade_date']).copy()

    all_dates = sorted(df['trade_date'].unique())
    all_codes = df['index_code'].unique()
    full_idx = pd.MultiIndex.from_product(
        [all_codes, all_dates], names=['index_code', 'trade_date']
    )
    df = (df.set_index(['index_code', 'trade_date'])
            .reindex(full_idx)
            .reset_index())
    df = df.sort_values(['index_code', 'trade_date'])

    df['flow_strength'] = df['flow_strength'].fillna(0)

    df['flow_cum'] = df.groupby('index_code')['flow_strength'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )
    df['flow_pct'] = df.groupby('trade_date')['flow_cum'].rank(pct=True)
    return df[['index_code', 'trade_date', 'flow_cum', 'flow_pct']].dropna(subset=['flow_cum'])


# ==================== 样本池 ====================
def build_universe(index_flow, index_daily, inst_ratio, mapping, index_name,
                   min_size=4e8, max_inst=0.70):
    index_daily = index_daily.copy()
    index_daily = _ensure_columns(index_daily, {
        'Trdmnt': 'trade_date', 'TradingDate': 'trade_date', 'date': 'trade_date',
        'IndexCode': 'index_code', 'indexcode': 'index_code', 'code': 'index_code',
        'Clsprc': 'close', 'ClosePrice': 'close', 'close_price': 'close', 'Clsindex': 'close',
    })
    index_daily['trade_date'] = pd.to_datetime(index_daily['trade_date'])
    index_daily['index_code'] = index_daily['index_code'].astype(str)
    index_daily = index_daily.sort_values(['index_code', 'trade_date'])
    if 'index_ret' not in index_daily.columns:
        index_daily['index_ret'] = index_daily.groupby('index_code')['close'].pct_change()

    inst_ratio = inst_ratio.copy()
    inst_ratio = _ensure_columns(inst_ratio, {
        'SecCode': 'etf_code', 'ETFCode': 'etf_code', 'code': 'etf_code', 'Symbol': 'etf_code',
        'ReportDate': 'report_date', 'EndDate': 'report_date', 'date': 'report_date',
        'InstRatio': 'inst_ratio', 'institution_ratio': 'inst_ratio',
        'inst_hold_ratio': 'inst_ratio', 'InstitutionRatio': 'inst_ratio',
        '机构占比': 'inst_ratio',
    })
    inst_ratio['report_date'] = pd.to_datetime(inst_ratio['report_date'])
    if 'available_date' not in inst_ratio.columns:
        inst_ratio['available_date'] = inst_ratio['report_date'] + pd.DateOffset(months=2)

    mapping = _ensure_columns(mapping, {
        'SecCode': 'etf_code', 'ETFCode': 'etf_code', 'code': 'etf_code',
        'IndexCode': 'index_code', 'indexcode': 'index_code', 'idx_code': 'index_code',
    })

    df = index_flow.merge(
        index_daily[['index_code', 'trade_date', 'index_ret']],
        on=['index_code', 'trade_date'], how='left'
    )

    inst_index = inst_ratio.merge(mapping[['etf_code', 'index_code']], on='etf_code', how='left')
    inst_index = inst_index.dropna(subset=['index_code'])
    inst_index = inst_index.groupby(['index_code', 'available_date']).agg(
        inst_ratio=('inst_ratio', 'mean')
    ).reset_index()

    df = df.sort_values(['index_code', 'trade_date']).reset_index(drop=True)
    inst_index = inst_index.sort_values(['index_code', 'available_date']).reset_index(drop=True)

    result = []
    for idx, g in df.groupby('index_code', sort=False):
        g = g.copy()
        g['index_code'] = idx  # groupby 后 index_code 变 index，手动补回
        sub = inst_index[inst_index['index_code'] == idx]
        if sub.empty:
            g['inst_ratio'] = np.nan
            g['available_date'] = pd.NaT
            result.append(g)
            continue
        g = pd.merge_asof(
            g.sort_values('trade_date'),
            sub.sort_values('available_date'),
            left_on='trade_date', right_on='available_date',
            direction='backward'
        )
        g['index_code'] = idx
        result.append(g)
    df = pd.concat(result, ignore_index=True)

    df = df.dropna(subset=['index_ret', 'inst_ratio'])
    df = df[df['prev_size'] > min_size]
    df = df[df['inst_ratio'] < max_inst]

    df = df.merge(index_name, on='index_code', how='left')
    return df


# ==================== 选股 ====================
def _select_holdings(pool, top_n, min_broad_n, broad_codes,
                     max_per_sector=2, name_map=None):
    name_map = name_map or {}
    pool = pool.sort_values('flow_pct')
    all_codes = pool['index_code'].tolist()
    sector_of = {c: _sector_tag(name_map.get(c, '')) for c in all_codes}

    selected = []
    sector_count = {}
    broad_selected = set()

    def try_add(code):
        s = sector_of.get(code, '其他')
        if s == '宽基':
            base = BROAD_DEDUP.get(code, code)
            if base in broad_selected:
                return False
        if sector_count.get(s, 0) >= max_per_sector:
            return False
        selected.append(code)
        sector_count[s] = sector_count.get(s, 0) + 1
        if s == '宽基':
            broad_selected.add(BROAD_DEDUP.get(code, code))
        return True

    broad_in_pool = [c for c in all_codes if c in broad_codes]
    for c in broad_in_pool:
        if sum(1 for x in selected if x in broad_codes) >= min_broad_n:
            break
        try_add(c)

    for c in all_codes:
        if len(selected) >= top_n:
            break
        if c in selected:
            continue
        try_add(c)

    return selected


# ==================== 回测 ====================
def backtest(universe, factor_df, top_n=5, freq='M', cost=0.001,
             min_broad_n=0, broad_codes=None, max_per_sector=2, name_map=None):
    broad_codes = broad_codes or set()
    name_map = name_map or {}
    df = universe.merge(
        factor_df[['index_code', 'trade_date', 'flow_cum', 'flow_pct']],
        on=['index_code', 'trade_date'], how='left'
    )
    df = df.dropna(subset=['flow_cum', 'index_ret']).copy()
    df = df.sort_values(['index_code', 'trade_date'])

    if freq == 'M':
        df['period'] = df['trade_date'].dt.to_period('M')
    else:
        df['period'] = df['trade_date'].dt.to_period('W')

    all_dates = pd.DatetimeIndex(sorted(df['trade_date'].unique()))
    cal = pd.DataFrame({'trade_date': all_dates})
    cal['period'] = cal['trade_date'].dt.to_period(freq)
    first_trade_dates = cal.groupby('period')['trade_date'].min()

    rebalance_periods = sorted(df['period'].unique())
    portfolio_ret = []
    prev_holdings = set()

    for i, period in enumerate(rebalance_periods):
        if period not in first_trade_dates.index:
            continue
        trade_date = first_trade_dates[period]

        cross = df[(df['period'] == period) & (df['trade_date'] == trade_date)].copy()
        if cross.empty:
            continue

        # 剔除资金流入：用截面分位 <= 0.5（相对阈值）
        neg = cross[cross['flow_pct'] <= 0.5]
        pool = neg if len(neg) >= top_n else cross

        if len(pool) < top_n:
            if prev_holdings:
                holdings = list(prev_holdings)
            else:
                continue
        else:
            holdings = _select_holdings(pool, top_n, min_broad_n, broad_codes,
                                        max_per_sector=max_per_sector,
                                        name_map=name_map)

        if i < len(rebalance_periods) - 1:
            next_period = rebalance_periods[i + 1]
            next_date = first_trade_dates.get(next_period, df['trade_date'].max())
            period_df = df[
                (df['trade_date'] > trade_date) &
                (df['trade_date'] <= next_date) &
                (df['index_code'].isin(holdings))
            ]
        else:
            period_df = df[
                (df['trade_date'] > trade_date) &
                (df['index_code'].isin(holdings))
            ]

        if len(period_df) > 0:
            pivot = period_df.pivot_table(
                index='trade_date', columns='index_code', values='index_ret'
            )
            pivot = pivot.reindex(columns=holdings).fillna(0)
            portfolio_ret.append(pivot.mean(axis=1))

        turnover = len(set(holdings) ^ prev_holdings) / max(len(holdings), 1) / 2.0
        if turnover > 0:
            portfolio_ret.append(pd.Series([-turnover * cost], index=[trade_date]))
        prev_holdings = set(holdings)

    if portfolio_ret:
        result = pd.concat(portfolio_ret).sort_index()
        return result[~result.index.duplicated(keep='last')]
    return pd.Series(dtype=float)


# ==================== 绩效 ====================
def evaluate(returns, benchmark_returns=None, freq=252):
    returns = returns.dropna()
    if len(returns) == 0:
        return {}
    nav = (1 + returns).cumprod()
    ann_ret = nav.iloc[-1] ** (freq / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).mean()
    result = {
        '年化收益': f'{ann_ret:.2%}', '年化波动': f'{ann_vol:.2%}',
        '夏普': f'{sharpe:.2f}', '最大回撤': f'{max_dd:.2%}',
        'Calmar': f'{calmar:.2f}', '胜率': f'{win_rate:.2%}'
    }
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).ffill()
        common = returns.index.intersection(bench.dropna().index)
        if len(common) > 0:
            excess = returns.loc[common] - bench.loc[common]
            ann_excess = (1 + excess).prod() ** (freq / len(excess)) - 1
            result['年化超额'] = f'{ann_excess:.2%}'
    return result


def yearly_performance(returns):
    df = pd.DataFrame({'ret': returns})
    df['year'] = df.index.year
    result = []
    for year, g in df.groupby('year'):
        if len(g) < 2:
            continue
        nav = (1 + g['ret']).cumprod()
        ann_ret = nav.iloc[-1] - 1
        dd = (nav / nav.cummax() - 1).min()
        vol = g['ret'].std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else np.nan
        result.append({'年份': year, '收益': f'{ann_ret:.2%}',
                       '最大回撤': f'{dd:.2%}', '夏普': f'{sharpe:.2f}'})
    return pd.DataFrame(result)


# ==================== 真实数据读取 ====================
def load_real_data():
    warnings.filterwarnings('ignore', message='Workbook contains no default style')

    print('\n[1/5] 读取CSMAR数据...')

    # 清洗 ETF 基础信息与指数映射
    etf_info = pd.read_excel(os.path.join(DATA_DIR, 'ETF.csv'))
    etf_info = etf_info.rename(columns={
        '证券代码': 'etf_code',
        '拟合指数': 'index_code',
        '当前规模(份)': 'shares_snapshot',
    })
    etf_info['etf_code'] = etf_info['etf_code'].astype(str).str.zfill(6)
    etf_info['index_code'] = etf_info['index_code'].apply(_clean_index_code)
    etf_info['shares_snapshot'] = (
        etf_info['shares_snapshot'].astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    mapping = etf_info[['etf_code', 'index_code']].drop_duplicates()
    mapping = mapping[mapping['index_code'].str.len() > 0]

    index_name = (etf_info[['index_code', '证券简称']]
                  .drop_duplicates()
                  .groupby('index_code')['证券简称']
                  .first()
                  .reset_index()
                  .rename(columns={'证券简称': 'index_name'}))

    print('  ETF 基本信息:', etf_info.shape,
          '| 唯一 ETF:', mapping['etf_code'].nunique(),
          '| 唯一指数:', mapping['index_code'].nunique())

    # 合并 ETF 日行情并处理份额缺失
    etf_files = sorted(glob.glob(os.path.join(DATA_DIR, 'FUND_MKT_Quotation*.csv')))
    print('  ETF 日行情文件:', [os.path.basename(f) for f in etf_files])
    etf_list = []
    for f in etf_files:
        df = pd.read_csv(f)
        df = df.rename(columns={
            'TradingDate': 'trade_date',
            'Symbol': 'etf_code',
            'ClosePrice': 'close',
            'CirculatedShare': 'shares',
        })
        etf_list.append(df)
    etf_daily = pd.concat(etf_list, ignore_index=True)

    etf_daily['trade_date'] = pd.to_datetime(etf_daily['trade_date'])
    etf_daily['etf_code'] = etf_daily['etf_code'].astype(str).str.zfill(6)
    etf_daily['close'] = pd.to_numeric(etf_daily['close'], errors='coerce')
    etf_daily['shares'] = pd.to_numeric(etf_daily['shares'], errors='coerce')

    etf_daily = etf_daily[etf_daily['etf_code'].isin(mapping['etf_code'])]
    etf_daily = etf_daily.drop_duplicates(subset=['etf_code', 'trade_date'], keep='last')
    etf_daily = etf_daily.sort_values(['etf_code', 'trade_date'])

    etf_daily['shares'] = etf_daily.groupby('etf_code')['shares'].ffill()
    etf_daily['close'] = etf_daily.groupby('etf_code')['close'].ffill()

    etf_daily = etf_daily[
        (etf_daily['trade_date'] >= START_DATE) &
        (etf_daily['trade_date'] <= END_DATE)
    ]
    print('  ETF 日行情:', etf_daily.shape,
          '| 日期:', etf_daily['trade_date'].min(), '~', etf_daily['trade_date'].max(),
          '| ETF 数:', etf_daily['etf_code'].nunique())

    etf_daily['share_chg'] = etf_daily.groupby('etf_code')['shares'].diff()
    etf_daily['etf_flow'] = etf_daily['share_chg'] * etf_daily['close']
    etf_daily['etf_size'] = etf_daily['shares'] * etf_daily['close']

    # 读取指数日行情
    index_daily = pd.read_csv(os.path.join(DATA_DIR, 'csmar_index_daily_clean.csv'),
                              parse_dates=['trade_date'], dtype={'index_code': str},
                              low_memory=False)
    index_daily['index_code'] = index_daily['index_code'].astype(str)
    index_daily = index_daily[
        (index_daily['trade_date'] >= START_DATE) &
        (index_daily['trade_date'] <= END_DATE)
    ]
    index_daily = index_daily.sort_values(['index_code', 'trade_date'])
    index_daily['index_ret'] = index_daily.groupby('index_code')['close'].pct_change()
    print('  指数日行情:', index_daily.shape,
          '| 指数数:', index_daily['index_code'].nunique(),
          '| 与 ETF 映射交集:', len(set(index_daily['index_code']) & set(mapping['index_code'])))

    # 处理机构占比
    inst_ratio = pd.read_csv(os.path.join(DATA_DIR, 'Fund_HolderStructure.csv'))
    inst_ratio = inst_ratio.rename(columns={
        'Symbol': 'etf_code',
        'EndDate': 'report_date',
        'InstitutionRatio': 'inst_ratio',
    })
    inst_ratio['etf_code'] = inst_ratio['etf_code'].astype(str).str.zfill(6)
    inst_ratio['report_date'] = pd.to_datetime(inst_ratio['report_date'])
    inst_ratio['inst_ratio'] = pd.to_numeric(inst_ratio['inst_ratio'], errors='coerce')
    if inst_ratio['inst_ratio'].max() > 1.5:
        inst_ratio['inst_ratio'] = inst_ratio['inst_ratio'] / 100

    inst_ratio = inst_ratio[inst_ratio['etf_code'].isin(mapping['etf_code'])]
    inst_ratio = inst_ratio.drop_duplicates(subset=['etf_code', 'report_date'], keep='last')
    inst_ratio['available_date'] = inst_ratio['report_date'] + pd.DateOffset(months=2)
    inst_ratio = inst_ratio.sort_values(['etf_code', 'report_date'])
    print('  机构占比:', inst_ratio.shape,
          '| ETF 数:', inst_ratio['etf_code'].nunique())

    # 基准对齐
    wind_path = os.path.join(DATA_DIR, 'wind_all_a.csv')
    if os.path.exists(wind_path):
        benchmark = pd.read_csv(wind_path, parse_dates=['trade_date'])
        benchmark = benchmark.sort_values('trade_date')
        benchmark['index_ret'] = benchmark['close'].pct_change()
        print('  基准: wind_all_a.csv')
    else:
        for code in ['000985', '000300']:
            sub = index_daily[index_daily['index_code'] == code][['trade_date', 'close']].copy()
            if len(sub) > 0:
                benchmark = sub
                benchmark['index_ret'] = benchmark['close'].pct_change()
                print(f'  基准: {code}（fallback）')
                break

    return etf_daily, index_daily, inst_ratio, mapping, index_name, benchmark


# ==================== 主流程 ====================
if __name__ == '__main__':
    print('=' * 60)
    print('ETF资金流反转效应 指数轮动策略 复现')
    print('=' * 60)

    etf_daily, index_daily, inst_ratio, mapping, index_name, benchmark = load_real_data()

    print('\n[2/5] 计算资金流强度因子...')
    index_flow = compute_index_flow(etf_daily, mapping)
    factor_df = compute_flow_factor(
        index_flow, window=FACTOR_WINDOW, min_periods=FACTOR_MIN_PERIODS
    )
    print('  因子记录:', factor_df.shape,
          '| 因子覆盖指数:', factor_df['index_code'].nunique())

    print('\n[3/5] 构建样本池...')
    universe = build_universe(
        index_flow, index_daily, inst_ratio, mapping, index_name,
        min_size=MIN_ETF_SIZE, max_inst=MAX_INST_RATIO
    )
    if len(universe) > 0:
        print('  样本池记录数:', len(universe))
        print('  样本池日期:', universe['trade_date'].min(), '~', universe['trade_date'].max())
        print('  样本池指数数:', universe['index_code'].nunique())
        print('  样本池日均指数数: {:.1f}'.format(
            universe.groupby('trade_date')['index_code'].nunique().mean()))
    else:
        print('  ⚠️ 样本池为空')

    print(f'\n[4/5] 回测 ({REBALANCE_FREQ}度调仓)...')
    name_map = dict(zip(index_name['index_code'], index_name['index_name']))
    returns = backtest(
        universe, factor_df,
        top_n=TOP_N, freq=REBALANCE_FREQ, cost=COST_RATE,
        min_broad_n=MIN_BROAD_N, broad_codes=BROAD_BASE_CODES,
        max_per_sector=MAX_PER_SECTOR, name_map=name_map
    )
    print(f'  策略收益序列: {len(returns)} 个交易日')

    print('\n[5/5] 绩效评估...')
    benchmark_ret = benchmark.set_index('trade_date')['index_ret']
    perf = evaluate(returns, benchmark_ret)
    print('\n全样本绩效：')
    for k, v in perf.items():
        print(f'  {k}: {v}')

    print('\n分年绩效：')
    yearly = yearly_performance(returns)
    if len(yearly) > 0:
        print(yearly.to_string(index=False))
    else:
        print('  无有效年度数据')

    print('\n' + '=' * 60)
    print('完成。')
    print('=' * 60)