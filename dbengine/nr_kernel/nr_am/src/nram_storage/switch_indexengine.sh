#!/bin/bash
# =============================================
# 切换 nrindex 索引引擎版本
# =============================================
# 用法:
#   ./switch_indexengine.sh single      # 切换到单线程版本 (原版 ALEX)
#   ./switch_indexengine.sh concurrent  # 切换到并发版本 (alexol)
#   ./switch_indexengine.sh status      # 查看当前使用的版本
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SINGLE_FILE="$SCRIPT_DIR/indexengine_single.cpp"
CONCURRENT_FILE="$SCRIPT_DIR/indexengine_concurrent.cpp"
ACTIVE_FILE="$SCRIPT_DIR/indexengine.cpp"

show_status() {
    echo "=============================================="
    echo "当前索引引擎版本:"
    echo "=============================================="

    if [ -f "$ACTIVE_FILE" ]; then
        if grep -q "alexol" "$ACTIVE_FILE" 2>/dev/null; then
            echo "  当前: 并发版本 (alexol/TBB)"
        elif grep -q "ALEX_CUR" "$ACTIVE_FILE" 2>/dev/null; then
            echo "  当前: 并发版本 (alexol/TBB)"
        else
            echo "  当前: 单线程版本 (原版 ALEX)"
        fi
    else
        echo "  错误: indexengine.cpp 不存在"
    fi

    echo ""
    echo "可用版本:"
    [ -f "$SINGLE_FILE" ] && echo "  - single (单线程)"
    [ -f "$CONCURRENT_FILE" ] && echo "  - concurrent (并发)"
    echo ""
}

switch_to_single() {
    echo "切换到单线程版本..."

    # 备份当前文件为并发版本（如果是并发版本的话）
    if [ -f "$ACTIVE_FILE" ] && grep -q "alexol\|ALEX_CUR" "$ACTIVE_FILE" 2>/dev/null; then
        cp "$ACTIVE_FILE" "$CONCURRENT_FILE"
        echo "  已备份当前并发版本到 indexengine_concurrent.cpp"
    fi

    # 如果存在单线程备份，恢复它
    if [ -f "$SINGLE_FILE" ]; then
        cp "$SINGLE_FILE" "$ACTIVE_FILE"
        echo "  已切换到单线程版本"
    else
        echo "  错误: indexengine_single.cpp 不存在"
        echo "  请先备份当前单线程版本"
        exit 1
    fi

    echo ""
    echo "请重新编译: make clean && make"
}

switch_to_concurrent() {
    echo "切换到并发版本..."

    # 备份当前文件为单线程版本（如果是单线程版本的话）
    if [ -f "$ACTIVE_FILE" ] && ! grep -q "alexol\|ALEX_CUR" "$ACTIVE_FILE" 2>/dev/null; then
        cp "$ACTIVE_FILE" "$SINGLE_FILE"
        echo "  已备份当前单线程版本到 indexengine_single.cpp"
    fi

    # 如果存在并发版本，恢复它
    if [ -f "$CONCURRENT_FILE" ]; then
        cp "$CONCURRENT_FILE" "$ACTIVE_FILE"
        echo "  已切换到并发版本"
    else
        echo "  错误: indexengine_concurrent.cpp 不存在"
        exit 1
    fi

    echo ""
    echo "请重新编译: make clean && make"
    echo ""
    echo "注意: 并发版本需要 TBB 库支持"
    echo "安装 TBB: sudo apt-get install libtbb-dev"
}

case "$1" in
    single)
        switch_to_single
        ;;
    concurrent)
        switch_to_concurrent
        ;;
    status|"")
        show_status
        ;;
    *)
        echo "用法: $0 {single|concurrent|status}"
        echo ""
        echo "  single      - 切换到单线程版本 (原版 ALEX)"
        echo "  concurrent  - 切换到并发版本 (alexol + TBB)"
        echo "  status      - 查看当前版本状态"
        exit 1
        ;;
esac
