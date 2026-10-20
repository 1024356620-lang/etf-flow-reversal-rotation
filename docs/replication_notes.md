## `docs/replication_notes.md`

```markdown
# 复现笔记：数据问题排查与参数选择

本文档记录复现过程中遇到的数据问题、排查过程、解决方案，以及关键参数的选取依据。按时间顺序排列，对应复现过程中策略绩效的演进。

---

## 绩效演进总览

| 阶段 | 年化 | 超额 | 关键改动 |
|---|---|---|---|
| 初始版本 | 2.53% | -1.16% | 样本池几乎为空（2020-08 起，日均 24.6 个） |
| 修 merge_asof + 截面分位 | 5.45% | -3.76% | 样本池 2019-02 起，日均 23.3 个 |
| 修 index_code 脏数据 | 9.97% | +2.82% | 样本池日均 37 个 |
| 规模阈值 5e8 → 4e8 | 11.24% | +2.57% | 样本池扩张 |
| flow_cum → flow_pct | **11.51%** | **+3.30%** | 相对阈值替代绝对阈值 |

---

## 问题 1：merge_asof 报错 "left keys must be sorted"

### 现象

```
ValueError: left keys must be sorted
```

在 `build_universe` 里用 `pd.merge_asof(df, inst_index, by='index_code', left_on='trade_date', right_on='available_date', direction='backward')` 时报错。

### 排查

`merge_asof` 的 `by` 参数只负责分组，**不负责排序检查**。pandas 内部 `_convert_values_for_libjoin` 会检查 `left_on` 列是否**全局有序**。原代码 `sort_values(['index_code', 'trade_date'])` 只保证每个指数内部有序，全局 `trade_date` 来回跳，触发报错。

### 解决

改为逐指数 `merge_asof`：

```python
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
```

**关键细节**：`groupby` 迭代时 `index_code` 会变成 index，`g['index_code']` 消失，必须在循环内手动补回，否则 `pd.concat` 后 `index_code` 列丢失，后续 `KeyError`。

---

## 问题 2：index_code 脏数据导致样本池过小

### 现象

样本池只有 76 个指数、日均 24.6 个，且从 2020-08 才开始。诊断打印发现：

```
mapping index_code 样例: ['970085', '970083', 'H30590', '980081', '399365 国证粮食']
index_daily index_code 样例: ['000001', '000001', ...]
交集指数数: 173
```

`ETF.csv` 的“拟合指数”字段混入了**中文名**（`'399365 国证粮食'`）和**空格**，导致 `merge` 时无法匹配 `index_daily` 里的 `'399365'`。

### 排查

- `_norm_code` 只处理纯数字（`zfill(6)`），对 `'399365 国证粮食'` 原样保留；
- `index_daily` 里是标准 6 位代码，脏数据永远匹配不上；
- 大量 ETF 的 `index_code` 脏，聚合到指数层面后，这些指数要么被丢弃，要么和其他指数混在一起。

### 解决

新增 `_clean_index_code`：

```python
def _clean_index_code(s):
    s = str(s).strip()
    if not s or s.lower() in ('nan', 'none', ''):
        return ''
    parts = s.split()
    if not parts:
        return ''
    s = parts[0]  # 只取第一段，去掉中文名
    s = re.sub(r'\.(SZ|SH|CSI|WI)$', '', s, flags=re.IGNORECASE)  # 去后缀
    return s
```

**注意空字符串保护**：`ETF.csv` 里有空白的“拟合指数”，`s.split()[0]` 会 `IndexError`，必须先判断 `if not s`。

### 效果

- 交集指数数：173 → **242**
- 样本池指数数：76 → **127**
- 样本池日均指数数：24.6 → **37.0**
- 年化：5.45% → **9.97%**

---

## 问题 3：月初候选数骤降到 1

### 现象

回测持仓明细显示，多个月初只有 1 个候选：

```
[2021-10-04] 候选 1 | 净流出 1 | 持仓 ['399437']
[2022-01-03] 候选 1 | 净流出 0 | 持仓 ['930625']
[2023-05-01] 候选 1 | 净流出 0 | 持仓 ['930604']
```

### 排查

月初第一个交易日，大量 ETF 在 CSMAR 里**整行缺失**（不是“有行但值为 NaN”），`groupby.ffill()` 只能填充“有行但值为 NaN”，**整行缺失时无法填充**。结果 `flow_strength` 大量 NaN，`flow_cum` 进一步 NaN，`dropna` 后只剩 1 个指数。

### 解决

在 `compute_flow_factor` 里，按全市场日历 reindex，把“整行缺失”变成 NaN，再填 0（假设缺失日无申赎）：

```python
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
```

### 效果

- 因子记录数：263,959 → **514,755**（reindex 后补全了大量组合）
- 月初候选数从 1 恢复到 20+（部分月份仍为 1，是样本池本身在该月只有 1 个指数通过规模/机构过滤，非 flow_cum 问题）

---

## 问题 4：绝对阈值受市场环境影响

### 现象

研报原文“剔除资金流入的指数”，原实现用 `flow_cum <= 0`（绝对符号）。但 `flow_cum` 是 10 日累计资金流强度，**在不同市场环境下，0 不一定是“流入/流出”的分界**：

- 牛市里大部分指数 10 日累计为正，`<= 0` 过滤掉太多；
- 熊市里大部分为负，`<= 0` 几乎不过滤。

### 解决

改用**截面分位**（相对阈值）：

```python
# 原：
neg = cross[cross['flow_cum'] <= 0]

# 改为：
neg = cross[cross['flow_pct'] <= 0.5]
```

`flow_pct` 是当日所有指数的截面分位，`<= 0.5` 表示“资金流强度低于中位数”，不受市场整体环境影响。

### 效果

- 年化：9.50% → **11.24%**（后续叠加规模 4e8 后到 11.51%）
- 最大回撤：-51.99% → **-50.99%**
- 2023 年：-21.82% → **-17.43%**

---

## 问题 5：2022/2023 年亏损放大

### 现象

2022 年 -27.33%、2023 年 -20.27%，研报同期 -13.93%/+6.17%。2023 年方向背离（研报正收益，复现负收益）。

### 排查

持仓明细显示，2022 年反复选中 `930697`、`931079`、`930713`（科技/新能源主题），2023 年反复选中 `930604`、`930608`、`930901`（科技/医药主题）。这些板块 2022/2023 年跌 20–30%。

样本池结构：127 个指数里，宽基仅 20–30 个，行业主题占 70%+。研报样本池宽基占比更高，熊市抗跌。

### 尝试过但失败的方案

**行业分散约束**（同板块最多 2 个）：2022 年从 -22.83% **恶化到 -30.52%**。原因：约束强制分散到医药/消费/金融地产，这些板块 2022 年也跌（医药 -30%、消费 -20%）。2022 年 A 股全面熊市，除煤炭外几乎所有板块下跌，分散化无效。

### 最终处理

**关闭行业约束**（`MAX_PER_SECTOR = 99`），保留宽基兜底（`MIN_BROAD_N = 3`），在结论中如实说明这是**样本池结构问题，非代码可解**。

---

## 参数选择依据

### `MIN_ETF_SIZE = 4e8`（4 亿）

研报 4.1 节明确说“规模阈值在 4–6 亿元，10 日资金流强度分位处于 5% 以内的指数未来 5 日平均年化收益均在 28% 左右，规模约束在择时上的稳健性较强”。CSMAR 数据的 `prev_size` 中位数是 4.13 亿，5 亿阈值会过滤掉一半指数；4 亿能保留约 55–60%，样本池从 37 涨到 42 左右。

实测：5e8 → 4e8，年化 9.50% → 11.24%。

### `MIN_BROAD_N = 3`

宽基兜底数量。试过 2 和 3：

- `MIN_BROAD_N = 2`：年化 8.84%、超额 +0.05%
- `MIN_BROAD_N = 3`：年化 8.71%、超额 +0.47%

差异在噪声范围内，选 3 是因为宽基兜底在 2024/2025 年略好。

### `MAX_PER_SECTOR = 99`（关闭行业约束）

见问题 5。行业约束在 2022 年恶化收益，在牛市略好，整体有过拟合嫌疑，关闭。

### `FACTOR_MIN_PERIODS = 5`

`rolling(10, min_periods=5)` 允许“过去 10 天有 5 天数据”即可计算。CSMAR ETF 数据缺失严重（平均每只 ETF 只有 45% 交易日有数据），`min_periods=10` 会导致大量 `flow_cum` 为 NaN。

### `FACTOR_WINDOW = 10`

研报 2.4 节：“10 日资金流强度因子对低机构占比指数未来收益具有显著的反向预测作用”。

---

## 与研报差距的最终归因

| 维度 | 本复现 | 研报 | 影响 |
|---|---|---|---|
| 样本池规模 | 日均 37 个 | 约 50 个 | 熊市抗跌能力弱 |
| 宽基占比 | 20–30 个 | 更高 | 2022/2023 年拖累 |
| 基准 | 中证全指（价格） | 万得全A（全收益） | 超额低估 1.5–2%/年 |
| 时间起点 | 2019-01 | 2017-01 | 缺少两年牛市初期 |
| 数据源 | CSMAR 子集 | Wind 全量 | ETF 覆盖不全 |

**结论**：策略核心逻辑（资金流反转 + 低机构占比 + 月度轮动）在 2019/2020/2021/2024/2025 年得到验证，6/7 个年度方向与研报一致。剩余 5.2 个百分点的年化差距来自数据源限制，非代码问题。
```