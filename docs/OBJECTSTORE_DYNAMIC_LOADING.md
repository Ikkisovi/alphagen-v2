# ObjectStore 动态因子加载方�?

**完成时间**: 2025-11-01
**状�?*: 已实�?

## 🎯 方案概述

使用 Lean Cloud ObjectStore 实现因子�?*动态热加载**，无需暂停实盘即可更新因子�?

### 核心优势

�?**无需暂停实盘** - 新因子在月初自动加载
�?**无缝切换** - 策略在运行时检测月份变�?
�?**完全自动�?* - 可通过 CI/CD 自动化整个流�?
�?**优雅降级** - 如果新因子不可用，继续使用旧因子

---

## 📋 架构说明

### 传统方案 vs ObjectStore 方案

| 特�?| 传统方案 | ObjectStore 方案 |
|------|----------|------------------|
| 策略部署 | 每月一个独立策�?| 单一策略，动态加�?|
| 月度切换 | 需要暂�?重新部署 | 自动切换，无需暂停 |
| 因子存储 | 硬编码在策略代码�?| 存储�?ObjectStore |
| 更新延迟 | 人工操作，有延迟 | 自动，午�?00:01 |
| 运维复杂�?| 高（手动切换�?| 低（全自动） |

### 工作流程

```
月末训练（实盘仍在运行）
    �?
导出 ObjectStore 格式
    �?
上传�?Lean Cloud ObjectStore
    �?
月初 00:01 自动加载新因�?
    �?
无缝切换，不影响实盘
```

---

## 🚀 快速开�?

### 1. 训练并导出因�?

```bash
# 完整训练 + 导出 ObjectStore 格式
python scripts/run_rolling_train.py --export-objectstore

# 或者只训练特定窗口
python scripts/run_rolling_train.py \
  --start-window 13 \
  --end-window 13 \
  --export-objectstore
```

**输出位置**: `lean_project/storage/factors/`

### 2. 上传�?ObjectStore

```bash
# 方法 1: 使用上传脚本（推荐）
python scripts/upload_factors_to_objectstore.py

# 方法 2: 手动上传
cd lean_project/storage/factors
lean cloud object-store set factors/2025_01.json --file 2025_01.json
lean cloud object-store set factors/2025_02.json --file 2025_02.json
```

### 3. 部署动态策�?

**首次部署**:
```bash
# 复制策略�?Lean 项目
cp lean_project/DynamicRollingStrategy.py <your-lean-project>/main.py

# 部署�?Lean Cloud
cd <your-lean-project>
lean cloud push
lean cloud live deploy
```

**后续更新**: 无需任何操作！策略会在每�?1 日自动加载新因子�?

---

## 📊 详细使用指南

### 导出 ObjectStore 格式

运行训练脚本时添�?`--export-objectstore` 参数�?

```bash
python scripts/run_rolling_train.py --export-objectstore
```

**生成的文件结�?*:
```
lean_project/storage/factors/
├── 2024_01.json
├── 2024_02.json
├── 2024_03.json
├── ...
├── 2025_10.json
└── manifest.json
```

**JSON 文件格式** (2024_01.json):
```json
{
  "version": "1.0",
  "deploy_month": "2024_01",
  "deploy_range": ["2024-01-01", "2024-01-31"],
  "window_idx": 12,
  "train_ic": 0.1234,
  "deploy_ic": 0.1100,
  "export_timestamp": "2025-11-01T10:30:00",
  "expressions": [
    "Mean($close, 20)",
    "Div($volume, $close)",
    ...
  ],
  "weights": [0.15, 0.12, ...],
  "n_factors": 10
}
```

### 上传�?ObjectStore

#### 使用上传脚本（推荐）

```bash
# 上传所有因�?
python scripts/upload_factors_to_objectstore.py

# 只上传特定月�?
python scripts/upload_factors_to_objectstore.py --month 2025_03

# 预览（不实际上传�?
python scripts/upload_factors_to_objectstore.py --dry-run

# 自定义因子目�?
python scripts/upload_factors_to_objectstore.py \
  --factors-dir /path/to/factors
```

**输出示例**:
```
================================================================================
Upload Factors to Lean Cloud ObjectStore
================================================================================
Factors directory: lean_project/storage/factors
================================================================================

Found 22 factor file(s) to upload

Uploading 2024_01.json...
  Source: lean_project/storage/factors/2024_01.json
  Destination: ObjectStore key 'factors/2024_01.json'
  Factors: 10
  Deploy month: 2024_01
  Train IC: 0.1234
  �?Uploaded successfully

...

================================================================================
Upload Summary
================================================================================
Total files: 22
Successful: 22
Failed: 0

�?Upload complete!
================================================================================
```

#### 手动上传

```bash
cd lean_project/storage/factors

# 上传单个文件
lean cloud object-store set factors/2025_01.json --file 2025_01.json

# 批量上传（bash�?
for f in *.json; do
  if [ "$f" != "manifest.json" ]; then
    lean cloud object-store set "factors/$f" --file "$f"
  fi
done
```

### 动态策略说�?

**核心机制**:

1. **月份检�?* - 每天 00:01 检查月份是否变�?
2. **因子加载** - 如果月份变化，从 ObjectStore 加载新因�?
3. **月度重平�?* - 每月第一个交易日重新构建组合

**关键代码片段**:

```python
def CheckAndUpdateFactors(self):
    """每天检查并更新因子"""
    current_month = self.Time.strftime("%Y_%m")

    # 月份变化�?
    if current_month == self.current_month:
        return

    # �?ObjectStore 加载新因�?
    factor_key = f"factors/{current_month}.json"

    if self.ObjectStore.ContainsKey(factor_key):
        factor_json = self.ObjectStore.Read(factor_key)
        factor_data = json.loads(factor_json)

        self.factor_expressions = factor_data['expressions']
        self.factor_weights = factor_data['weights']
        self.current_month = current_month

        self.Log(f"�?Loaded {len(self.factor_expressions)} factors for {current_month}")
```

---

## 🔄 实盘运维流程

### 月度操作时间�?

**2�?8�?晚上**（实盘运行中�?

```bash
# 1. 训练 3 月因子（~1-2小时，取决于配置�?
python scripts/run_rolling_train.py \
  --start-window 14 \
  --end-window 14 \
  --export-objectstore

# 2. 上传�?ObjectStore（~1分钟�?
python scripts/upload_factors_to_objectstore.py --month 2025_03
```

**3�?�?00:01**（自动发生）:

```
策略自动执行 CheckAndUpdateFactors()
  �?
检测到月份�?"2025_02" 变为 "2025_03"
  �?
�?ObjectStore 加载 factors/2025_03.json
  �?
更新因子表达式和权重
  �?
记录日志：✅ Loaded 10 factors for 2025_03
```

**3�?�?09:30**（市场开盘后30分钟�?

```
策略执行 Rebalance()
  �?
使用新加载的 3 月因子计算信�?
  �?
构建新的投资组合
  �?
执行交易
```

### 故障处理

**场景 1: 新因子未及时上传**

```python
# 策略日志
⚠️  Factors for 2025_03 not found in ObjectStore
   Continuing with factors from 2025_02
```

**处理**: 补充上传后，策略会在第二�?00:01 自动加载�?

**场景 2: 因子文件损坏**

```python
# 策略日志
�?Error loading factors for 2025_03: JSONDecodeError
   Continuing with factors from 2025_02
```

**处理**: 修复 JSON 并重新上传�?

**场景 3: 首次启动无因�?*

```python
# 策略日志
⚠️  No factors available, strategy will not trade
   Please upload factors to: factors/2025_03.json
```

**处理**: 上传当月因子后重启策略，或等待第二天自动加载�?

---

## 🤖 自动化方�?

### GitHub Actions 自动�?

创建 `.github/workflows/monthly_retrain.yml`:

```yaml
name: Monthly Factor Retraining and Upload

on:
  schedule:
    # 每月 28 �?20:00 UTC
    - cron: '0 20 28 * *'
  workflow_dispatch:  # 支持手动触发

jobs:
  retrain-and-upload:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install lean

      - name: Calculate next month window
        id: window
        run: |
          # Calculate which window to train
          NEXT_MONTH=$(date -d "next month" +%Y_%m)
          WINDOW_IDX=$(($(date +%m) + 12))
          echo "next_month=$NEXT_MONTH" >> $GITHUB_OUTPUT
          echo "window_idx=$WINDOW_IDX" >> $GITHUB_OUTPUT

      - name: Train next month factors
        run: |
          python scripts/run_rolling_train.py \
            --start-window ${{ steps.window.outputs.window_idx }} \
            --end-window ${{ steps.window.outputs.window_idx }} \
            --export-objectstore

      - name: Configure Lean CLI
        run: |
          lean login --user-id ${{ secrets.LEAN_USER_ID }} \
                     --api-token ${{ secrets.LEAN_API_TOKEN }}

      - name: Upload to ObjectStore
        run: |
          python scripts/upload_factors_to_objectstore.py \
            --month ${{ steps.window.outputs.next_month }}

      - name: Send notification
        if: success()
        run: |
          echo "�?Factors for ${{ steps.window.outputs.next_month }} uploaded successfully"
          # 可以添加 Slack/Email 通知
```

### Cron Job 自动化（Linux/Mac�?

```bash
# 编辑 crontab
crontab -e

# 添加月度任务（每�?28 �?20:00�?
0 20 28 * * cd /path/to/alphagen && /path/to/venv/bin/python scripts/run_rolling_train.py --start-window $(date +\%m --date="next month") --end-window $(date +\%m --date="next month") --export-objectstore && /path/to/venv/bin/python scripts/upload_factors_to_objectstore.py --month $(date +\%Y_\%m --date="next month")
```

### Windows Task Scheduler

1. 打开任务计划程序
2. 创建基本任务
3. 触发器：每月
4. 操作：运行脚�?
   ```batch
   C:\Python39\python.exe E:\factor\alphagen\scripts\run_rolling_train.py --export-objectstore
   ```
5. 添加后续操作：上传脚�?

---

## 📁 文件结构


```
alphagen/
├── scripts/
�?  ├── run_rolling_train.py                # 添加�?--export-objectstore
�?  └── upload_factors_to_objectstore.py    # 新增：上传脚�?
├── alphagen_lean/
�?  └── lean_exporter.py                    # 添加�?ObjectStore 导出方法
├── lean_project/
�?  ├── storage/
�?  │   └── factors/                        # ObjectStore 因子（新格式�?
�?  │       ├── 2024_01.json
�?  │       ├── 2024_02.json
�?  │       └── manifest.json
�?  └── DynamicRollingStrategy.py           # 新增：动态加载策略模�?
├── output/
�?  └── rolling_results/                    # 训练结果（传统格式）
└── OBJECTSTORE_DYNAMIC_LOADING.md          # 本文�?
```

---

## 🔍 验证和测�?

### 本地测试

```bash
# 1. 训练单窗�?+ 导出
python scripts/run_rolling_train.py \
  --end-window 1 \
  --export-objectstore \
  --no-export

# 2. 检查输�?
cat lean_project/storage/factors/2024_01.json | jq '.'

# 3. 预览上传
python scripts/upload_factors_to_objectstore.py --dry-run
```

### ObjectStore 验证

```bash
# 列出所有因�?
lean cloud object-store list --prefix factors/

# 查看特定因子
lean cloud object-store get factors/2024_01.json
```

### 策略日志监控

�?Lean Cloud 中查看策略日志：

```
2025-03-01 00:01:00 : Month changed to 2025_03, loading factors from factors/2025_03.json
2025-03-01 00:01:01 : �?Loaded 10 factors for 2025_03
2025-03-01 00:01:01 :    Train IC: 0.1234, Deploy IC: 0.1100
2025-03-01 09:30:00 : Monthly Rebalance - 2025-03-01
2025-03-01 09:30:00 : Using factors from: 2025_03
```

---

## ⚙️ 配置选项

### run_rolling_train.py 新参�?

```bash
--export-objectstore    # 导出 ObjectStore 格式（除了常规导出）
```

### upload_factors_to_objectstore.py 参数

```bash
--factors-dir PATH      # 因子目录（默认：lean_project/storage/factors�?
--month YYYY_MM         # 只上传指定月�?
--dry-run              # 预览模式，不实际上传
```

### DynamicRollingStrategy.py 配置

可以在策略中修改�?

```python
# ObjectStore 配置
self.objectstore_prefix = "factors/"  # 可改为自定义前缀

# 组合参数
self.lookback_days = 60
self.top_quantile = 0.2
self.max_position_size = 0.15
self.max_position_count = 20

# 风控参数
self.min_dollar_volume = 1000000
self.min_price = 5.0
```

---

## 🎯 最佳实�?

### 1. 提前训练

在月�?*至少提前 1 �?*训练和上传新因子，留有缓冲时间�?

```bash
# 2�?7日就可以训练 3 月因�?
python scripts/run_rolling_train.py --start-window 14 --end-window 14 --export-objectstore
```

### 2. 监控日志

重点监控�?
- �?月初 00:01 的因子加载日�?
- �?月初 09:30 的重平衡日志
- ⚠️ 任何警告或错误信�?

### 3. 备份因子

```bash
# 上传前备�?
cp -r lean_project/storage/factors lean_project/storage/factors_backup_$(date +%Y%m%d)
```

### 4. 渐进式部�?

**首次使用建议**:

1. 先在回测中测试动态策�?
2. 用纸上交易验证月度切�?
3. 确认无误后再部署到实�?

### 5. 版本管理

```bash
# 给因子打标签
git tag -a factors-v1.0 -m "Initial factor release"
git push origin factors-v1.0
```

---

## �?FAQ

**Q1: ObjectStore 有大小限制吗�?*
A: 单个文件最�?5MB。我们的因子 JSON 通常只有�?KB，完全没问题�?

**Q2: 如果训练失败了怎么办？**
A: 策略会继续使用上个月的因子，不影响实盘运行�?

**Q3: 可以手动触发因子更新吗？**
A: 可以通过 Lean Cloud API 发送命令让策略重新调用 `CheckAndUpdateFactors()`�?

**Q4: 因子计算逻辑在哪里？**
A: 当前模板中是 placeholder。需要集�?`expression_converter.py` 将表达式转为实际计算代码�?

**Q5: 支持多策略吗�?*
A: 支持。每个策略可以从 ObjectStore 的不同前缀读取�?
```python
# 策略 A
self.objectstore_prefix = "factors_strategyA/"
# 策略 B
self.objectstore_prefix = "factors_strategyB/"
```

---

## 🔗 相关文档

- [ROLLING_UPGRADE_COMPLETE.md](ROLLING_UPGRADE_COMPLETE.md) - 滚动训练系统升级
- [DAILY_DATA_SUPPORT.md](DAILY_DATA_SUPPORT.md) - 日线数据支持
- [QUICK_START.md](QUICK_START.md) - 快速开始指�?

---

## �?完成清单

- [x] 添加 ObjectStore 导出方法�?`lean_exporter.py`
- [x] 修改 `run_rolling_train.py` 支持 `--export-objectstore`
- [x] 创建 `DynamicRollingStrategy.py` 动态加载策略模�?
- [x] 创建 `upload_factors_to_objectstore.py` 上传脚本
- [x] 编写完整使用文档

---

**下一�?*: 运行完整训练并部署到实盘�?

```bash
# 1. 训练所有窗�?
python scripts/run_rolling_train.py --export-objectstore

# 2. 上传所有因�?
python scripts/upload_factors_to_objectstore.py

# 3. 部署策略
cd <your-lean-project>
cp ../alphagen/lean_project/DynamicRollingStrategy.py main.py
lean cloud push
lean cloud live deploy
```

🎉 完成！实盘现在支持动态因子加载，月度自动切换�?
