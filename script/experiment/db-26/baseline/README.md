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
