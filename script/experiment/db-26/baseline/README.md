# Baseline inference

## PREDICT 怎么调用 Python？

1. **Postgres 端**（`dbengine/nr_kernel/nr_pipeline` + `nr_ext`）
   - 解析 `PREDICT CLASS OF click_rate FROM (SELECT * FROM frappe_extend) TRAIN ON *`
   - 先查 `router` 表：`(table_name, feature_columns_hash, target_columns_hash)` → 若有则得到 `model_id`，走推理；否则走训练。
   - **训练**：只连一个 AI engine（`engines[0]`），通过 **WebSocket** 发 `T_TRAIN` 任务和逐批数据（libsvm 格式）；Python 训练完后 `insert_model` + `register_model` 写回 DB。
   - **推理**：多 engine 时建多个 WebSocket，每个发 `T_INFERENCE` 任务；每批数据按行数均分给各 worker，各自从 DB `load_model(model_id)` 再算，结果在 C 端合并。

2. **Python 端**（`aiengine/runtime/server.py`）
   - 起 Quart + WebSocket，监听 `/ws`。
   - 收到 `batch_data`：把数据放进 `DataCache`，由 `LibSvmDataDispatcher` 预处理成 batch。
   - 收到 `inference` 任务：`inference_task(setup, model_id, inf_batch_num, feature_names, target_name, session_id)` → `setup.inference()` 里用 `conn.load_model(model_id).unpack().to_model()` 加载模型，再从 `StreamingDataSet` 里按批取数据，调用 `builder.inference()`（如 armnet 的 forward），结果通过 WebSocket 流式回传。

3. **模型存哪**
   - 在 **Postgres** 里：`model`（model_id, model_meta）、`layer`（model_id, layer_id, layer_data）、`router`（model_id, table_name, feature_columns, target_columns）。

## Baseline 脚本做什么？

`baseline_inference.py` 是一个**不经过 WebSocket、不经过 PREDICT 管线**的对照实现：

- 用 **SQL** 查 `router` 得到 `model_id`（与 C 端相同的 hash 规则）。
- 用 **NeurDB storeman** 的 `load_model(model_id).unpack().to_model()` 从 DB 读模型（与 algserver 一致）。
- 用 **SQL** 按 batch 读特征列（默认表中除 target 外的全部列）。
- 在 Python 里按批拼成 armnet 需要的 `{id, value}`，调用 `model(batch)` 做推理。

用来和「完整 PREDICT 路径」比延迟和正确性。

## 怎么跑

1. 先保证库里已有对应模型（对目标表跑过一次带 `TRAIN ON *` 的 PREDICT）。
2. 在仓库根目录下执行：

```bash
# 默认表 frappe_test，目标 click_rate，特征为除 click_rate 外的全部列
python script/experiment/db-26/baseline/baseline_inference.py

# 指定表和目标
python script/experiment/db-26/baseline/baseline_inference.py --table frappe_extend --target click_rate

# 限制 batch 数、限制行数、写出预测结果
python script/experiment/db-26/baseline/baseline_inference.py --table frappe_test --target click_rate --num-batches 10 --limit 6000 --out preds.txt
```

DB 连接默认 `localhost:5432`、库/用户 `neurdb`，可用 `--db-host`、`--db-port`、`--db-name`、`--db-user` 覆盖。

## 分布式 inference 实验流程

1. **环境**
   - 推荐在容器里跑：`docker exec -it neurdb_dev bash`，进入 `/code/neurdb-dev`。
   - 宿主机如需跑 Python baseline，先切到有依赖的环境：`conda activate neurbench`。

2. **准备数据和模型**
   - `frappe_test` 默认由 `test/frappe.csv` 导入，`test/test.sh` 会建表、`COPY` 数据，并执行一次 `PREDICT ... TRAIN ON *`。
   - 表结构是 `click_rate + feature1..feature10`。后续系统 inference 和 baseline 都默认读 `frappe_test`。

3. **启动多 worker AI engine**
   ```bash
   ./script/ai_servers/start_ai_servers.sh 3
   ```
   - 这个脚本会启动 `8090,8091,8092` 三个 Python server，日志写到 `test/server_*.log`。
   - 按脚本输出，在 psql 里重新注册 engine：
     ```sql
     DELETE FROM pg_catalog.nr_aiengine;
     select insert_ai_engine('127.0.0.1', 8090);
     select insert_ai_engine('127.0.0.1', 8091);
     select insert_ai_engine('127.0.0.1', 8092);
     ```
   - 并行度就是注册的 engine 数量。

4. **跑系统路径**
   - 通过 `PREDICT CLASS/VALUE OF click_rate FROM frappe_test TRAIN ON *;` 触发系统 inference。
   - C 端按注册的 engine 数把数据切分，多个 Python worker 分别加载同一个 model_id 并推理。

5. **跑 baseline**
   ```bash
   python script/experiment/db-26/baseline/baseline_inference.py
   python script/experiment/db-26/baseline/baseline_inference.py --num-batches 10
   ```
   - baseline 不走 WebSocket，只是 Python 直连 DB：查 router、加载模型、按 batch 读表并调用 `model(batch)`。
   - 脚本日志会输出 `load_ms`、`infer_ms`、`total_ms`。

6. **收集日志**
   - 系统路径看 `test/server_8090.log`、`test/server_8091.log`、`test/server_8092.log`。
   - 重点关注：`loading model from database`、`done batch for ...`、batch 数、每个 server 的总耗时。
   - baseline 直接看终端输出即可。

7. **后续分析**
   - 当前还没有正式画图脚本；先手工记录不同 worker 数、不同 feature 数、不同 batch 数下的 `total_ms/infer_ms`。
   - 后续可以补一个小脚本解析 `server_*.log` 和 baseline 输出，汇总成 CSV，再画折线图或柱状图。
