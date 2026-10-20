# 数据说明

本项目代码依赖以下 CSMAR 数据文件。由于商业数据版权限制，原始数据不随仓库分发，请自行通过 CSMAR 数据库下载并放入项目根目录的 `raw_data/` 文件夹。

## 必需文件

| 文件名 | 内容 | CSMAR 数据库模块 |
|---|---|---|
| `ETF.csv` | ETF 基本信息（代码、拟合指数、证券简称、当前规模） | 基金研究系列 / 基金基本信息 |
| `FUND_MKT_Quotation*.csv` | ETF 日行情（收盘价、流通份额） | 基金市场系列 / 基金行情 |
| `csmar_index_daily_clean.csv` | 指数日行情（指数代码、交易日、收盘价） | 股票市场系列 / 指数日行情 |
| `Fund_HolderStructure.csv` | ETF 持有人结构（机构占比） | 基金研究系列 / 基金持有人结构 |

## 目录结构要求

```text
etf-flow-reversal-rotation/
├── raw_data/                 <-- 把上述 CSV 放到这里
│   ├── ETF.csv
│   ├── FUND_MKT_Quotation.csv
│   ├── ...
│   ├── csmar_index_daily_clean.csv
│   └── Fund_HolderStructure.csv
├── src/
│   └── strategy.py
└── ...