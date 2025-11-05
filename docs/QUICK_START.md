# Quick Start - 滚动训练升级版

## 🚀 一键运行

### 完整滚动训练 (2023-2025)
```bash
cd e:/factor/alphagen

# 双阶段训练 + 日线数据 (推荐)
python scripts/run_rolling_train.py
```

就这么简单！默认使用：
- ✅ 日线数据 (daily)
- ✅ 双阶段训练 (dual_stage)
- ✅ 6000步/阶段

## ⚡ 快速测试

### 单窗口测试 (~2分钟)
```bash
python scripts/run_rolling_train.py \
  --end-window 1 \
  --price-steps 500 \
  --fundamental-steps 500 \
  --no-export
```

### 3窗口测试 (~10分钟)
```bash
python scripts/run_rolling_train.py \
  --end-window 3 \
  --price-steps 2000 \
  --fundamental-steps 2000 \
  --no-export
```

## 📊 查看结果

```bash
# 训练摘要
cat output/rolling_results/training_summary.json | jq '.windows[] | {window: .deploy_month, ic: .deploy_ic}'

# IC 分数列表
cat output/rolling_results/training_summary.json | jq '.windows[].deploy_ic'

# 平均 IC
cat output/rolling_results/training_summary.json | jq '[.windows[].deploy_ic] | add/length'
```

## 🎛️ 常用选项

| 选项 | 默认 | 说明 |
|------|------|------|
| --resolution | daily | minute 或 daily |
| --train-strategy | dual_stage | single 或 dual_stage |
| --price-steps | 6000 | 价格阶段步数 |
| --fundamental-steps | 6000 | 基本面阶段步数 |
| --end-window | 全部 | 训练窗口数量 |
| --device | cpu | cpu 或 cuda:0 |

## 📁 输出位置

- **训练结果**: `output/rolling_results/`
- **Lean 策略**: `lean_project/strategies/`

## 🐛 故障排除

### 问题: 日期超出范围
```bash
# 修改配置中的起始日期
# rolling_config.py: first_train_start = "2023-01-03"  # 改为交易日
```

### 问题: 内存不足
```bash
# 减少窗口数或使用更少步数
python scripts/run_rolling_train.py --end-window 5 --price-steps 3000 --fundamental-steps 3000
```

## 📚 更多信息

- 详细文档: [ROLLING_UPGRADE_COMPLETE.md](ROLLING_UPGRADE_COMPLETE.md)
- 日线数据: [DAILY_DATA_QUICKSTART.md](DAILY_DATA_QUICKSTART.md)
- 帮助: `python scripts/run_rolling_train.py --help`

---

**升级完成时间**: 2025-11-01 ✅
