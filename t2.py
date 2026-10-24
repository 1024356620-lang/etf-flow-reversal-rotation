import pandas as pd
for name in ['results_A', 'results_B', 'results_C']:
    a = pd.read_csv(f'{name}/account.csv', parse_dates=['trade_date']).set_index('trade_date')
    print(f'{name}: 平均仓位 {1 - a["cash_weight"].mean():.2%}, 年化 {((1 + a["net_ret"]).prod() ** (252 / len(a)) - 1):.2%}')