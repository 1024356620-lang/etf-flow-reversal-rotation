# ETF 资金流反转效应指数轮动策略复现与回测验证

## 项目简介
独立复现中泰证券《指数应用研究系列二：基于ETF资金流反转效应与持有人结构异质性的指数轮动策略》（2026-03-15）。
本项目不仅完成了策略的月度调仓复现，还从零构建了账本自洽的回测引擎，并比较了绝对阈值、相对排序与仓位匹配等权对照在历史样本中的表现。回测区间为 2019-01-02 至 2026-01-30（2026 年结果仅含 1 月）。

## 核心工程贡献
1. **构建自洽的回测账本**：修复了持仓估值污染、成本覆盖收益、每日重新等权等回测 Bug；实现每日资产权重+现金权重严格为 1。
2. **信号时点验证**：通过所列信号时点与截断一致性检查，验证了 T-1 信号与当日截面分位的对齐关系，并排查了缺失前日原始因子的边界情况。
3. **缺失收益与成本处理**：交易成本只对证券买卖计费；实现缺失价格的前向填充估值。
   *注：调仓日旧持仓收益缺失时，按零收益记账，即沿用此前账面估值计算退出；这属于回测估值假设，不代表存在可成交报价。非调仓日采用价格前向填充估值，恢复报价后的收益衔接仍需结合原始行情核验。*

## A/B/C 历史表现比较
统一使用 T-1 信号、月度调仓、日频估值，对比了三种方案：

- **A（绝对净流出 TOP5）**：年化 **10.43%**，最大回撤 -49.04%，平均仓位 90.74%
  *规则：最多 5 个仓位，受宽基配额与缺额留现金约束。*
- **B（相对资金流 TOP5）**：年化 **13.92%**，最大回撤 -53.57%，平均仓位 98.72%
  *规则：不设绝对流出阈值，其余同 A。*
- **C（同仓位全池等权）**：年化 **7.12%**，最大回撤 -48.00%，平均仓位 90.74%
  *规则：在调仓日匹配 A 的目标总仓位，等权配置于候选池，随后自然漂移。*

**研究发现与结论**：
1. **调仓目标仓位匹配下的表现差异**：A（10.43%）相对 C（7.12%）具有约 3.31 个百分点的年化收益差。这支持 A 的整套筛选与配置规则在该历史样本中优于候选池等权，但需注意该差异也可能包含宽基配额等规则的贡献，尚不能单独归因为资金流因子。
2. **绝对阈值与仓位暴露**：B 的年化收益高于 A，同时平均仓位也更高，因此收益差异可能同时来自仓位暴露与持仓选择，当前实验尚未分离两者的贡献。2022 年 A、B 的年度收益分别为 -18.86% 和 -20.80%，A 在该年度亏损较小，但单个年度不足以证明绝对流出阈值具有稳定的熊市防守效果。

> **结果生成环境**：
> - 代码版本：Commit `c40432f6f5fda774d99276576cdd384588460871`
> - 配置：`flow_filter_mode` (absolute/relative/c_equal_weight), `min_etf_size=5e8`, `top_n=5`, `signal_lag='t1'`, `cost_rate=0.001`。

## 复现方法

### 1. 环境依赖
```bash
pip install -r requirements.txt
# 依赖：pandas>=2.0, numpy>=1.24, openpyxl>=3.1, pytest>=7.0
```

### 2. 数据准备
将以下 CSMAR 数据文件放入 `raw_data/` 目录：
- `ETF.csv`：ETF 基本信息（包含代码、拟合指数等）
- `FUND_MKT_Quotation*.csv`：ETF 日行情（收盘价、流通份额）
- `csmar_index_daily_clean.csv`：指数日行情（代码、日期、收盘价）
- `Fund_HolderStructure.csv`：ETF 持有人结构（机构占比）

### 3. 执行命令（T-1 保守信号，月度调仓）
```bash
# 运行 A 方案（绝对净流出）
python src/strategy.py --flow-filter-mode absolute --min-etf-size 5e8 --signal-lag t1 --out-dir results_A

# 运行 B 方案（相对资金流）
python src/strategy.py --flow-filter-mode relative --min-etf-size 5e8 --signal-lag t1 --out-dir results_B

# 运行 C 方案（同仓位等权）
python src/strategy.py --flow-filter-mode c_equal_weight --min-etf-size 5e8 --signal-lag t1 --out-dir results_C

# 运行 t0 方案（用于信号时点校验的对照）
python src/strategy.py --flow-filter-mode absolute --min-etf-size 5e8 --signal-lag t0 --out-dir results_t0
```

### 4. 核验脚本
```bash
# 核验回测账本内部一致性
python tests/verify_account.py --results-dir results_A

# 验证信号时点与原始因子一致性（使用生成的 t0 和 A 的 t1 结果）
python tests/verify_signal_lag.py --t0 results_t0 --t1 results_A

# 核验缺失收益的溯源标签
python tests/check_missing_returns.py --results-dir results_C
```

## 免责声明
本项目的历史回测结果基于 2019-2026 历史区间，可能受特定样本期与市场 Beta 影响，不构成投资建议。策略的后续优化应在预先划定的样本外区间进行评价。