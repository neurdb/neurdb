#!/bin/bash
# =============================================
# pgbench 点查询测试: NRINDEX vs BTREE 不行就使用这个
# =============================================
# 测试表:   covid_nrindex (NRINDEX索引), covid_btree (BTREE索引)
# 测试Key:  从 query_keys 表随机选取
# 查询SQL:  SELECT * FROM table WHERE val = ? LIMIT 1
# =============================================

PSQL="/code/neurdb-dev/psql/bin/psql -h 127.0.0.1 -d neurdb"
PGBENCH="/code/neurdb-dev/psql/bin/pgbench -h 127.0.0.1"
DBNAME="neurdb"

# 测试参数
CLIENTS=64          # 并发连接数
THREADS=10          # 线程数
DURATION=60         # 测试时长(秒)
PROGRESS=5          # 进度输出间隔(秒)

echo "=============================================="
echo "pgbench 点查询测试: NRINDEX vs BTREE"
echo "=============================================="
echo "测试内容: 随机点查询 SELECT * FROM table WHERE val = ? LIMIT 1"
echo "并发连接: $CLIENTS"
echo "线程数: $THREADS"
echo "测试时长: ${DURATION}s"
echo "=============================================="

# =============================================
# Step 1: 检查表是否存在
# =============================================
echo ""
echo "Step 1: 检查表是否存在..."

check_table() {
    local table=$1
    local exists=$($PSQL -t -A -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '$table';")
    echo "$exists"
}

COVID_EXISTS=$(check_table "covid")
COVID_NRINDEX_EXISTS=$(check_table "covid_nrindex")
COVID_BTREE_EXISTS=$(check_table "covid_btree")
QUERY_KEYS_EXISTS=$(check_table "query_keys")

echo "  covid:         $([ "$COVID_EXISTS" -eq 1 ] && echo '存在' || echo '不存在')"
echo "  covid_nrindex: $([ "$COVID_NRINDEX_EXISTS" -eq 1 ] && echo '存在' || echo '不存在')"
echo "  covid_btree:   $([ "$COVID_BTREE_EXISTS" -eq 1 ] && echo '存在' || echo '不存在')"
echo "  query_keys:    $([ "$QUERY_KEYS_EXISTS" -eq 1 ] && echo '存在' || echo '不存在')"

# 检查必要的表
if [ "$COVID_EXISTS" -eq 0 ]; then
    echo "错误: covid 表不存在，请先导入数据"
    exit 1
fi

if [ "$QUERY_KEYS_EXISTS" -eq 0 ]; then
    echo "错误: query_keys 表不存在，请先导入查询键"
    exit 1
fi

# =============================================
# Step 2: 显示数据量
# =============================================
echo ""
echo "Step 2: 数据量统计..."

COVID_COUNT=$($PSQL -t -A -c "SELECT COUNT(*) FROM covid;")
QUERY_KEYS_COUNT=$($PSQL -t -A -c "SELECT COUNT(*) FROM query_keys;")

echo "  covid 表:      $COVID_COUNT 行"
echo "  query_keys 表: $QUERY_KEYS_COUNT 行"

if [ "$COVID_NRINDEX_EXISTS" -eq 1 ]; then
    NRINDEX_COUNT=$($PSQL -t -A -c "SELECT COUNT(*) FROM covid_nrindex;")
    echo "  covid_nrindex: $NRINDEX_COUNT 行"
fi

if [ "$COVID_BTREE_EXISTS" -eq 1 ]; then
    BTREE_COUNT=$($PSQL -t -A -c "SELECT COUNT(*) FROM covid_btree;")
    echo "  covid_btree:   $BTREE_COUNT 行"
fi

# =============================================
# Step 3: 检查/创建测试表（仅在不存在时创建）
# =============================================
echo ""
echo "Step 3: 检查测试表..."

if [ "$COVID_NRINDEX_EXISTS" -eq 0 ]; then
    echo "  创建 covid_nrindex 表..."
    $PSQL << 'EOF'
CREATE TABLE covid_nrindex (id INT PRIMARY KEY, val BIGINT);
INSERT INTO covid_nrindex SELECT id, val FROM covid;
ANALYZE covid_nrindex;
EOF
    echo "  covid_nrindex 表已创建"
else
    echo "  covid_nrindex 表已存在，跳过创建"
fi

if [ "$COVID_BTREE_EXISTS" -eq 0 ]; then
    echo "  创建 covid_btree 表..."
    $PSQL << 'EOF'
CREATE TABLE covid_btree (id INT PRIMARY KEY, val BIGINT);
INSERT INTO covid_btree SELECT id, val FROM covid;
ANALYZE covid_btree;
EOF
    echo "  covid_btree 表已创建"
else
    echo "  covid_btree 表已存在，跳过创建"
fi

# =============================================
# Step 4: 重建索引
# =============================================
echo ""
echo "Step 4: 重建索引..."

# 删除旧索引
$PSQL -c "DROP INDEX IF EXISTS idx_nrindex;" 2>/dev/null
$PSQL -c "DROP INDEX IF EXISTS idx_btree;" 2>/dev/null

echo "  创建 NRINDEX 索引..."
NRINDEX_START=$(date +%s%3N)
$PSQL -c "CREATE INDEX idx_nrindex ON covid_nrindex USING nrindex(val);" 2>&1
NRINDEX_END=$(date +%s%3N)
NRINDEX_TIME=$((NRINDEX_END - NRINDEX_START))
echo "  idx_nrindex 创建完成 (${NRINDEX_TIME}ms)"

echo "  创建 BTREE 索引..."
BTREE_START=$(date +%s%3N)
$PSQL -c "CREATE INDEX idx_btree ON covid_btree USING btree(val);" 2>&1
BTREE_END=$(date +%s%3N)
BTREE_TIME=$((BTREE_END - BTREE_START))
echo "  idx_btree 创建完成 (${BTREE_TIME}ms)"

# =============================================
# Step 5: 准备查询键表
# =============================================
echo ""
echo "Step 5: 准备查询键表..."

$PSQL << 'EOF'
DROP TABLE IF EXISTS qk_idx;
CREATE UNLOGGED TABLE qk_idx (id INT PRIMARY KEY, val BIGINT);
INSERT INTO qk_idx SELECT ROW_NUMBER() OVER ()::INT, val FROM query_keys;
ANALYZE qk_idx;
EOF

KEY_COUNT=$($PSQL -t -A -c "SELECT COUNT(*) FROM qk_idx;")
echo "  qk_idx 表已创建: $KEY_COUNT 个查询键"

# =============================================
# Step 6: 创建测试 SQL 文件
# =============================================
echo ""
echo "Step 6: 创建测试 SQL 文件..."

cat > /tmp/bench_nrindex.sql << EOF
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
\set kid random(1, $KEY_COUNT)
SELECT * FROM covid_nrindex WHERE val = (SELECT val FROM qk_idx WHERE id = :kid) LIMIT 1;
EOF

cat > /tmp/bench_btree.sql << EOF
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
\set kid random(1, $KEY_COUNT)
SELECT * FROM covid_btree WHERE val = (SELECT val FROM qk_idx WHERE id = :kid) LIMIT 1;
EOF

echo "  /tmp/bench_nrindex.sql"
echo "  /tmp/bench_btree.sql"

# =============================================
# Step 7: 测试 NRINDEX
# =============================================
echo ""
echo "=============================================="
echo "Step 7: 测试 NRINDEX"
echo "=============================================="
NRINDEX_RESULT=$($PGBENCH -c $CLIENTS -j $THREADS -T $DURATION -P $PROGRESS -n -f /tmp/bench_nrindex.sql $DBNAME 2>&1)
echo "$NRINDEX_RESULT"

NRINDEX_TPS=$(echo "$NRINDEX_RESULT" | grep -oP 'tps = \K[0-9.]+' | head -1)
NRINDEX_LAT=$(echo "$NRINDEX_RESULT" | grep -oP 'latency average = \K[0-9.]+')

# =============================================
# Step 8: 测试 BTREE
# =============================================
echo ""
echo "=============================================="
echo "Step 8: 测试 BTREE"
echo "=============================================="
BTREE_RESULT=$($PGBENCH -c $CLIENTS -j $THREADS -T $DURATION -P $PROGRESS -n -f /tmp/bench_btree.sql $DBNAME 2>&1)
echo "$BTREE_RESULT"

BTREE_TPS=$(echo "$BTREE_RESULT" | grep -oP 'tps = \K[0-9.]+' | head -1)
BTREE_LAT=$(echo "$BTREE_RESULT" | grep -oP 'latency average = \K[0-9.]+')

# =============================================
# Step 9: 结果对比
# =============================================
echo ""
echo "=============================================="
echo "测试结果对比"
echo "=============================================="
echo ""
echo "索引创建时间:"
echo "  NRINDEX: ${NRINDEX_TIME}ms"
echo "  BTREE:   ${BTREE_TIME}ms"
echo ""
printf "%-10s | %-15s | %-15s\n" "索引" "TPS" "平均延迟(ms)"
printf "%-10s-+-%-15s-+-%-15s\n" "----------" "---------------" "---------------"
printf "%-10s | %-15s | %-15s\n" "NRINDEX" "$NRINDEX_TPS" "$NRINDEX_LAT"
printf "%-10s | %-15s | %-15s\n" "BTREE" "$BTREE_TPS" "$BTREE_LAT"

if [ -n "$NRINDEX_TPS" ] && [ -n "$BTREE_TPS" ]; then
    SPEEDUP=$(awk "BEGIN {if ($BTREE_TPS > 0) printf \"%.2f\", $NRINDEX_TPS / $BTREE_TPS; else print \"N/A\"}")
    echo ""
    echo "NRINDEX 相比 BTREE 提升: ${SPEEDUP}x"
fi

# 清理
rm -f /tmp/bench_nrindex.sql /tmp/bench_btree.sql

echo ""
echo "=============================================="
echo "测试完成!"
echo "=============================================="
