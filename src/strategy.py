# src/strategy.py
import sys
from pathlib import Path

# 允许 python src/strategy.py 直接运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import replace

from config import StrategyConfig
from pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow-filter-mode', default='absolute',
                        choices=['absolute', 'percentile'])
    parser.add_argument('--inst-lag-months', type=int, default=2)
    parser.add_argument('--cost-rate', type=float, default=0.001)
    parser.add_argument('--min-etf-size', type=float, default=4e8)
    args = parser.parse_args()

    config = replace(StrategyConfig(),
                     flow_filter_mode=args.flow_filter_mode,
                     inst_lag_months=args.inst_lag_months,
                     cost_rate=args.cost_rate,
                     min_etf_size=args.min_etf_size)
    result = run(config)
    print('\n全样本绩效：')
    for k, v in result['performance'].items():
        print(f'  {k}: {v}')
    print('\n分年绩效：')
    print(result['yearly'].to_string(index=False))


if __name__ == '__main__':
    main()