# 日线数据快速开始指南

## 🚀 快速使用

### 1. 检查现有日线数据
```bash
cd /Data/equity/usa/daily
ls *.zip | head -20
```

当前已有 **62个股票** 的日线数据。

### 2. 下载缺失的股票数据（如需要）
```bash
# 查看你需要的股票列表
cat lean_project/output/fundamental_stage.json | grep -A 50 "ticker_pool"

# 下载单个股票
lean data download --dataset "US Equity Security Master" --resolution daily --ticker MU

# 批量下载
lean data download --dataset "US Equity Security Master" --resolution daily \
  --ticker MU TTMI CDE KGC COMM STRL DXPE WLDN SSRM LRN
```

### 3. 运行测试验证
```bash
cd lean_project

# 基础数据加载测试
python test_daily_data.py

# 完整管道测试（含 StockData 转换）
python test_full_pipeline.py
```

### 4. 使用日线数据训练

#### 选项 A: 价格特征 + 基本面特征（两阶段）
```bash
# 阶段1: 仅价格特征
python train_for_lean.py \
  --ticker-pool '["MU","CDE","ALL","CCL","APP","PYPL","WFC","BRK.B","TMUS","UBER"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution daily \
  --price-only-stage \
  --steps 6000 \
  --pool-capacity 10 \
  --output output/price_stage_daily.json

# 阶段2: 添加基本面特征
python train_for_lean.py \
  --ticker-pool '["MU","CDE","ALL","CCL","APP","PYPL","WFC","BRK.B","TMUS","UBER"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution daily \
  --warm-start output/price_stage_daily.json \
  --steps 6000 \
  --pool-capacity 10 \
  --output output/fundamental_stage_daily.json
```

#### 选项 B: 仅价格特征（单阶段）
```bash
python train_for_lean.py \
  --ticker-pool '["MU","CDE","ALL","CCL","APP"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution daily \
  --price-only-stage \
  --steps 6000 \
  --output output/factors_daily.json
```

#### 选项 C: 所有特征（单阶段）
```bash
python train_for_lean.py \
  --ticker-pool '["MU","CDE","ALL"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution daily \
  --steps 6000 \
  --output output/factors_daily_full.json
```

## 📊 性能优势

| 指标 | 分钟数据 | 日线数据 |
|------|---------|---------|
| 加载时间 | 10-30秒 | <1秒 |
| 内存占用 | ~数GB | ~数MB |
| 聚合时间 | 5-10秒 | 跳过 |
| **总节省** | - | **~30x 更快** |

## 🔍 验证结果

### 检查训练输出
```bash
# 查看生成的因子
cat output/factors_daily.json | jq '.expressions'

# 查看 IC 分数
cat output/factors_daily.json | jq '{train_ic, test_ic, test_ric}'

# 查看元数据
cat output/factors_daily.json | jq '.metadata'
```

### 对比分钟数据结果
```bash
# 训练两个版本
python train_for_lean.py --resolution minute ... > results_minute.txt
python train_for_lean.py --resolution daily ... > results_daily.txt

# 对比 IC 分数
diff results_minute.txt results_daily.txt
```

预期：IC 分数应该非常接近（差异 <5%）。

## ⚙️ 命令行参数

### 关键参数
```bash
--resolution {minute,daily}  # 数据分辨率，默认 daily
--ticker-pool JSON           # 股票列表，JSON数组格式
--start-date YYYY-MM-DD      # 开始日期
--end-date YYYY-MM-DD        # 结束日期
--steps INT                  # PPO训练步数，默认 5000
--pool-capacity INT          # 因子池容量，默认 10
--output PATH                # 输出JSON文件路径
--price-only-stage           # 仅使用价格特征（OHLCV）
--warm-start PATH            # 从已有因子热启动
```

### 完整参数列表
```bash
python train_for_lean.py --help
```

## 📁 数据路径结构

```
/Data/equity/usa/
├── daily/              ← 日线数据（推荐）
│   ├── mu.zip
│   ├── cde.zip
│   └── ...
└── minute/             ← 分钟数据（向后兼容）
    ├── mu/
    │   ├── 20230103_trade.zip
    │   ├── 20230104_trade.zip
    │   └── ...
    └── ...
```

## 🐛 常见问题

### Q: "Daily data zip not found"
**A**: 运行 `lean data download --resolution daily --ticker <SYMBOL>`

### Q: 如何知道哪些股票已有日线数据？
**A**:
```bash
ls /Data/equity/usa/daily/*.zip | sed 's/.*\///' | sed 's/\.zip//' | tr '\n' ','
```

### Q: 日线数据和分钟数据聚合的结果一样吗？
**A**: 应该非常接近。Lean 的日线数据是从分钟数据聚合的，使用相同的逻辑。

### Q: VWAP 为什么是近似值？
**A**: 日线数据通常不包含成交量加权平均价，代码使用收盘价作为近似。对于日线因子，这通常是可接受的。

## 📚 相关文档

- [DAILY_DATA_SUPPORT.md](DAILY_DATA_SUPPORT.md) - 详细技术文档
- [DAILY_DATA_UPDATE_SUMMARY.md](DAILY_DATA_UPDATE_SUMMARY.md) - 更新摘要
- [test_daily_data.py](lean_project/test_daily_data.py) - 测试脚本

## 💡 最佳实践

1. **始终使用日线数据**（除非需要日内信号）
2. **两阶段训练**获得更好的因子分离
3. **验证数据**在训练前运行测试脚本
4. **对比结果**与分钟数据版本进行比较
5. **监控 IC 分数**确保数据质量

## ✅ 检查清单

使用日线数据前：
- [ ] 确认日线数据已下载
- [ ] 运行 `test_daily_data.py` 验证
- [ ] 检查 ticker pool 中所有股票都有数据
- [ ] 确认日期范围在数据覆盖范围内
- [ ] 准备基本面数据（如需要）

训练后：
- [ ] 检查 IC 分数是否合理
- [ ] 验证因子表达式
- [ ] 与之前结果对比
- [ ] 在 Lean 中回测验证

---

**更新日期**: 2025-11-01
**测试状态**: ✅ 全部通过
**推荐使用**: 🚀 是
