import pandas as pd

# 读取 A 和 C 的账户记录与调仓明细
a_acct = pd.read_csv('results_A/account.csv', parse_dates=['trade_date']).set_index('trade_date')
c_acct = pd.read_csv('results_C/account.csv', parse_dates=['trade_date']).set_index('trade_date')

c_rb = pd.read_csv('results_C/rebalance_detail.csv', parse_dates=['trade_date'])
rb_dates = c_rb['trade_date']

# 提取调仓日的现金权重
m = a_acct[['cash_weight']].join(c_acct[['cash_weight']], lsuffix='_A', rsuffix='_C')
m_rb = m.loc[m.index.intersection(rb_dates)]

# 计算调仓日现金权重的绝对差异
diff = (m_rb['cash_weight_A'] - m_rb['cash_weight_C']).abs()

print(f'调仓日样本数: {len(diff)}')
print(f'调仓日 cash 最大差异: {diff.max():.3e}')
print(f'差异 > 1e-10 的次数: {(diff > 1e-10).sum()}')

# 抽查最近 5 个调仓日
print("\n最近 5 个调仓日现金权重对比:")
print(m_rb.tail(5)[['cash_weight_A', 'cash_weight_C']])