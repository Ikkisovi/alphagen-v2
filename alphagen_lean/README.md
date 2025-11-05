# AlphaGen-Lean Integration

自动化的Alpha因子挖掘与回测系统，结合AlphaGen的强化学习训练与Lean的回测框架。

## 功能特性

- **滚动窗口训练**：自动按月滚动训练Alpha因子
- **表达式转换**：将AlphaGen表达式自动转换为Python代码
- **Lean集成**：自动生成完整的Lean回测策略
- **横截面标准化**：与AlphaGen训练时保持一致的因子标准化
- **性能跟踪**：跟踪每个窗口的IC和RankIC指标

## 目录结构

```
alphagen_lean/
├── __init__.py                   # 模块初始化
├── rolling_config.py             # 配置类
├── data_prep.py                  # 数据加载与准备
├── window_manager.py             # 时间窗口管理
├── rolling_trainer.py            # 滚动训练主逻辑
├── expression_converter.py       # 表达式转换器
├── lean_exporter.py              # Lean策略导出
├── templates/                     # Lean策略模板
│   ├── main.py.template
│   ├── config.py.template
│   ├── factor_calculator.py.template
│   ├── data_aggregator.py.template
│   └── portfolio_constructor.py.template
└── README.md                      # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
# AlphaGen依赖已包含在项目中
pip install pandas numpy torch stable-baselines3 sb3-contrib
```

### 2. 配置参数

编辑 `alphagen_lean/rolling_config.py`：

```python
# 数据路径
data_path = r"E:\factor\lean_project\data\equity\usa\minute"

# 股票池
symbols = ['MU', 'TTMI', 'CDE', ...]

# 时间窗口
first_train_start = '2023-01-01'  # 训练起始日期
deploy_start = '2024-01-01'       # 部署起始日期
deploy_end = '2025-10-31'          # 部署结束日期
train_months = 12                  # 训练窗口：12个月
test_months = 1                    # 部署窗口：1个月

# 训练参数
pool_capacity = 10        # 因子池容量
train_steps = 10000       # 训练步数
forward_horizon = 20      # 预测天数
```

### 3. 运行滚动训练

```bash
# 训练所有窗口
python scripts/run_rolling_train.py

# 训练前3个窗口（用于快速测试）
python scripts/run_rolling_train.py --end-window 3

# 使用更少步数进行快速测试
python scripts/run_rolling_train.py --steps 1000 --end-window 1

# 仅导出已训练的结果
python scripts/run_rolling_train.py --export-only

# 仅训练，不导出
python scripts/run_rolling_train.py --no-export
```

### 4. 回测Lean策略

```bash
# 进入某个窗口的策略目录
cd lean_project/strategies/window_2024_01

# 运行回测
lean backtest

# 查看结果
cat results.json
```

## 工作流程

### 整体流程

```
Data Loading → Window Generation → Rolling Training → Expression Conversion → Lean Export → Backtest
```

### 详细步骤

1. **数据准备** (`data_prep.py`)
   - 从Lean分钟数据文件加载OHLCV
   - 聚合为日度数据
   - 保存为pickle供AlphaGen使用

2. **窗口管理** (`window_manager.py`)
   - 生成滚动时间窗口
   - 例：Window 0: Train(2023-01~2024-01) → Deploy(2024-02)

3. **滚动训练** (`rolling_trainer.py`)
   - 对每个窗口：
     - 切片数据
     - 初始化AlphaPool
     - 使用PPO训练
     - 保存最佳因子组合和权重

4. **表达式转换** (`expression_converter.py`)
   - 解析AlphaGen表达式树
   - 生成NumPy/Pandas代码
   - 例：`Mean($close, 20d)` → `np.mean(h['close'][-20:])`

5. **Lean导出** (`lean_exporter.py`)
   - 生成`config.py`（因子表达式+权重）
   - 生成`factor_calculator.py`（计算逻辑）
   - 复制其他模块（main.py等）

6. **回测验证**
   - 在Lean中运行策略
   - 验证实盘表现

## 配置说明

### 时间窗口参数

- `first_train_start`: 第一个训练窗口的开始日期（需要更早的数据用于warmup）
- `deploy_start`: 第一个部署窗口的开始日期
- `deploy_end`: 最后一个部署窗口的结束日期
- `train_months`: 训练窗口大小（月）
- `test_months`: 部署窗口大小（月）
- `step_months`: 滚动步长（月）

### AlphaGen训练参数

- `pool_capacity`: Alpha池容量（保留多少个因子）
- `train_steps`: PPO训练总步数
- `forward_horizon`: 前瞻收益天数（预测N天后收益）
- `max_backtrack`: 表达式最大回看天数
- `l1_alpha`: L1正则化强度

### Lean策略参数

- `long_short_mode`: True=多空策略，False=只做多
- `top_quantile`: 做多比例（前20%）
- `max_position_size`: 单仓位最大占比（15%）
- `min_dollar_volume`: 最小成交额过滤（$1M）

## 输出结构

```
output/rolling_results/
├── rolling_config.json          # 配置备份
├── all_windows_data.pkl         # 全部窗口数据
├── training_summary.json        # 训练汇总
├── window_2024_01/
│   ├── window_info.json
│   ├── final_report.json       # IC、RankIC等指标
│   ├── pool_states/            # 每个rollout的pool状态
│   ├── checkpoints/            # PPO模型checkpoints
│   └── tensorboard/            # TensorBoard日志
└── window_2024_02/
    └── ...

lean_project/strategies/
├── index.json                   # 策略索引
├── window_2024_01/
│   ├── main.py
│   ├── config.py               # 因子表达式+权重
│   ├── factor_calculator.py    # _f1()~_fN()函数
│   ├── data_aggregator.py
│   └── portfolio_constructor.py
└── window_2024_02/
    └── ...
```

## 支持的算子

### 基础算子
- **特征**: $close, $open, $high, $low, $volume, $vwap
- **常量**: 数值常量（如 10.0, -0.5）
- **二元运算**: Add, Sub, Mul, Div, Pow, Greater, Less
- **一元运算**: Abs, Log, Sign, Sqrt

### 滚动算子
- **统计量**: Mean, Sum, Std, Var, Max, Min
- **时序**: Delta, Ref
- **高级**: Mad, WMA, EMA, Corr, Cov, Rank

### 示例表达式

```python
# 20日均线
"Mean($close, 20d)"

# 价格相对于10日均线的位置
"Div($close, Mean($close, 10d))"

# 成交量动量
"Div(Mean($volume, 5d), Mean($volume, 20d))"

# 复杂表达式
"Mul(Div(Mean(Mad(Div(10.0,Greater($high,-0.01)),40d),40d),-2.0),1.0)"
```

## 重要配置说明

### Deploy IC 与预测窗口模式

- `rolling_config.py` 默认 `evaluate_deploy = True`，当部署窗口有数据时会自动计算 Deploy IC / RankIC；若部署区间为空（例如预测未来月份），系统会打印 `Skipping deploy evaluation (no deploy data available)` 并在 `final_report.json` 中写入 `null`，但训练和因子输出照常进行。
- 需要生成未来预测窗口时，将 `deploy_end` 推到目标月份即可（当前配置已延长到 `2025-11-30`，因此会生成 `window_2025_11`）；该窗口的 Deploy 指标为 `null`，Lean 导出的策略仍可用于前瞻测试。
- 层叠窗口训练完成后，通过 `python scripts/run_rolling_train.py --start-window <idx> --end-window <idx+1>` 可单独重训任意窗口，导出的 `config.py` 会携带准确的 `START_DATE`/`END_DATE` 供 Lean 预热。
- 表达式转换器已经补充 `Med(...)` 支持，若仍遇到不支持的算子会在导出时 fallback 为 `np.nan` 并输出警告。

### 数据源路径配置 ⚠️

**关键配置**（[rolling_config.py:20](rolling_config.py#L20)）：
```python
_base_data_path: Path = Path(r"E:\factor\alphagen\lean_project\data\equity\usa")
```

**路径验证**：
- ✅ 正确：`E:\factor\alphagen\lean_project\data\equity\usa`（项目内数据）
- ❌ 错误：`E:\factor\lean_project\data\equity\usa`（外层目录）

**常见问题症状**：
- 数据日期范围不完整
- Deploy IC异常为0
- 训练数据量少于预期

**排查步骤**：
1. 检查`rolling_config.py`中的`_base_data_path`配置
2. 确认路径存在：`ls lean_project/data/equity/usa/daily/*.zip | head`
3. 清除缓存：`rm output/rolling_results/all_windows_data.pkl`
4. 重新训练验证

### 因子训练结果json数据源
结果在E:\factor\alphagen\output\rolling_results然后export to the E:\factor\alphagen\lean_project\storage
### BRK.B符号特殊处理

ZIP文件中的CSV文件名可能不一致（`brk.b.csv` vs `brkb.csv`）。
已在[data_prep.py:127-150](data_prep.py#L127-L150)实现多种文件名尝试逻辑。

## 常见问题

### Q: 训练很慢怎么办？
A:
- 减少`train_steps`（例如从10000降到2000）
- 减少`pool_capacity`
- 使用GPU：设置`device='cuda:0'`
- 减少训练窗口数量

### Q: 表达式转换失败怎么办？
A:
- 检查`expression_converter.py`是否支持该算子
- 如果不支持，会生成返回NaN的fallback函数
- 可以手动在`factor_calculator.py`中修复

### Q: Lean回测结果与预期IC不符？
A:
- 确保横截面标准化逻辑正确
- 检查数据对齐（时间戳、symbol列表）
- 验证因子计算逻辑（可以添加debug输出）

### Q: 内存不足？
A:
- 减少股票数量
- 减少数据时间范围
- 使用分批处理

## 进阶用法

### 自定义配置

```python
from alphagen_lean.rolling_config import RollingConfig

config = RollingConfig()
config.train_months = 6      # 使用6个月训练窗口
config.pool_capacity = 15    # 保留15个因子
config.device = 'cuda:0'     # 使用GPU

# 使用自定义配置
from alphagen_lean.rolling_trainer import RollingTrainer
trainer = RollingTrainer(config)
results = trainer.train_all_windows()
```

### 部分窗口重训练

```python
# 只重训练window 5到window 10
trainer.train_all_windows(start_window=5, end_window=10)
```

### 导出特定窗口

```python
from alphagen_lean.lean_exporter import LeanExporter

exporter = LeanExporter(config)
exporter.export_window(window_result, output_dir)
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 参考资料

- [AlphaGen GitHub](https://github.com/RL-MLDM/alphagen)
- [QuantConnect Lean Documentation](https://www.quantconnect.com/docs)
- AlphaGen论文: "Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning" (KDD 2023)
