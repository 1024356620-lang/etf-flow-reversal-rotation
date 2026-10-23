import pandas as pd

diag = pd.read_csv('results/diagnostics.csv', parse_dates=['trade_date'])
diag['year'] = diag['trade_date'].dt.year

print(diag.groupby('year').agg(
    n_months=('trade_date', 'count'),
    n_broad_ok=('broad_ok', 'sum'),
    n_broad_fail=('broad_ok', lambda x: (~x).sum()),
    avg_broad_n=('broad_n', 'mean'),
    avg_selected=('n_selected', 'mean'),
))