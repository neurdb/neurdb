#!/bin/bash

# ==============================================
# 数据构造脚本 - 用于索引性能测试
# 直接从已有CSV文件导入数据
# ==============================================

set -e

# 配置参数 - 已有的数据文件路径
BULK_LOAD_FILE="/hdd9/benjamin/adaptive-learned-index/result_figure_revison/covid_bulk_load.csv"
QUERY_KEY_FILE="/hdd9/benjamin/adaptive-learned-index/result_figure_revison/covid_query_key.csv"
INSERT_KEY_FILE="/hdd9/benjamin/adaptive-learned-index/result_figure_revison/covid_insert_key.csv"

BULK_LOAD_COUNT=20000000
QUERY_KEY_COUNT=20000000
INSERT_KEY_COUNT=10000000

DB_NAME="neurdb"

# ==============================================
# 步骤1: 创建数据表
# ==============================================
echo "=== 步骤1: 创建数据表 ==="

psql -d ${DB_NAME} << 'SQLEOF'
-- 删除已存在的表
DROP TABLE IF EXISTS covid_nrindex CASCADE;
DROP TABLE IF EXISTS covid_btree CASCADE;
DROP TABLE IF EXISTS query_keys CASCADE;
DROP TABLE IF EXISTS insert_keys CASCADE;

-- 创建 covid_nrindex 表 (用于nrindex索引测试)
CREATE TABLE covid_nrindex (
    id INT PRIMARY KEY,
    val BIGINT
);

-- 创建 covid_btree 表 (用于btree索引对比测试)
CREATE TABLE covid_btree (
    id INT PRIMARY KEY,
    val BIGINT
);

-- 创建 query_keys 表 (存储查询用的key)
CREATE TABLE query_keys (
    id INT PRIMARY KEY,
    val BIGINT
);

-- 创建 insert_keys 表 (存储待插入的key)
CREATE TABLE insert_keys (
    id INT PRIMARY KEY,
    val BIGINT
);

\echo '表创建完成'
SQLEOF

# ==============================================
# 步骤2: 导入数据
# ==============================================
echo ""
echo "=== 步骤2: 导入数据 ==="

# 使用临时表导入，然后用 row_number 生成 id
echo "导入 bulk_load 数据到 covid_nrindex (${BULK_LOAD_COUNT} 条)..."
psql -d ${DB_NAME} << SQLEOF
CREATE TEMP TABLE tmp_load (val BIGINT);
\COPY tmp_load(val) FROM '${BULK_LOAD_FILE}' WITH (FORMAT csv, HEADER true)
INSERT INTO covid_nrindex (id, val) SELECT row_number() OVER ()::INT, val FROM tmp_load;
DROP TABLE tmp_load;
SQLEOF

echo "导入 bulk_load 数据到 covid_btree (${BULK_LOAD_COUNT} 条)..."
psql -d ${DB_NAME} << SQLEOF
CREATE TEMP TABLE tmp_load (val BIGINT);
\COPY tmp_load(val) FROM '${BULK_LOAD_FILE}' WITH (FORMAT csv, HEADER true)
INSERT INTO covid_btree (id, val) SELECT row_number() OVER ()::INT, val FROM tmp_load;
DROP TABLE tmp_load;
SQLEOF

echo "导入 query_key 数据到 query_keys (${QUERY_KEY_COUNT} 条)..."
psql -d ${DB_NAME} << SQLEOF
CREATE TEMP TABLE tmp_load (val BIGINT);
\COPY tmp_load(val) FROM '${QUERY_KEY_FILE}' WITH (FORMAT csv, HEADER true)
INSERT INTO query_keys (id, val) SELECT row_number() OVER ()::INT, val FROM tmp_load;
DROP TABLE tmp_load;
SQLEOF

echo "导入 insert_key 数据到 insert_keys (${INSERT_KEY_COUNT} 条)..."
psql -d ${DB_NAME} << SQLEOF
CREATE TEMP TABLE tmp_load (val BIGINT);
\COPY tmp_load(val) FROM '${INSERT_KEY_FILE}' WITH (FORMAT csv, HEADER true)
INSERT INTO insert_keys (id, val) SELECT row_number() OVER ()::INT, val FROM tmp_load;
DROP TABLE tmp_load;
SQLEOF

echo "验证导入的数据:"
psql -d ${DB_NAME} -c "SELECT 'covid_nrindex' as table_name, count(*) FROM covid_nrindex UNION ALL SELECT 'covid_btree', count(*) FROM covid_btree UNION ALL SELECT 'query_keys', count(*) FROM query_keys UNION ALL SELECT 'insert_keys', count(*) FROM insert_keys;"

# ==============================================
# 步骤3: 创建索引
# ==============================================
echo ""
echo "=== 步骤3: 创建索引 ==="

echo "在 covid_nrindex 上创建 nrindex 索引..."
psql -d ${DB_NAME} -c "CREATE INDEX idx_covid_nrindex ON covid_nrindex USING nrindex(val);"

echo "在 covid_btree 上创建 btree 索引..."
psql -d ${DB_NAME} -c "CREATE INDEX idx_covid_btree ON covid_btree USING btree(val);"

echo ""
echo "=== 数据准备完成 ==="
echo "- covid_nrindex: ${BULK_LOAD_COUNT} 条记录，已创建 nrindex 索引"
echo "- covid_btree: ${BULK_LOAD_COUNT} 条记录，已创建 btree 索引"
echo "- query_keys: ${QUERY_KEY_COUNT} 条查询key"
echo "- insert_keys: ${INSERT_KEY_COUNT} 条待插入key"
