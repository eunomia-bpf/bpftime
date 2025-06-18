#!/bin/bash

# 正确的 Bloom Filter 测试脚本
# 类似 queue_demo 的测试方式：Server 监控，Client 触发事件
set -e

echo "=========================================="
echo "  正确的 Bloom Filter 测试演示"
echo "=========================================="
echo ""

# 获取绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查文件
if [ ! -f "./uprobe_bloom_filter" ] || [ ! -f "./target" ]; then
    echo "[错误] 请先运行 'make' 构建程序"
    exit 1
fi

# 使用绝对路径
UPROBE_PROGRAM="$SCRIPT_DIR/uprobe_bloom_filter"
TARGET_PROGRAM="$SCRIPT_DIR/target"
SYSCALL_SERVER_LIB="$SCRIPT_DIR/../../build/runtime/syscall-server/libbpftime-syscall-server.so"
AGENT_LIB="$SCRIPT_DIR/../../build/runtime/agent/libbpftime-agent.so"

if [ ! -f "$SYSCALL_SERVER_LIB" ] || [ ! -f "$AGENT_LIB" ]; then
    echo "[错误] bpftime 库文件不存在，请先构建 bpftime"
    echo "  期望位置: $SYSCALL_SERVER_LIB"
    echo "  期望位置: $AGENT_LIB"
    exit 1
fi

echo "测试原理说明："
echo "  类似 queue_demo 的测试方式："
echo "  1. Server: 监控程序运行，等待 uprobe 事件"
echo "  2. Client: 目标程序运行，触发函数调用"
echo "  3. 验证: Server 通过 bloom filter 分析用户访问模式"
echo ""

# 清理函数
cleanup() {
    echo ""
    echo "清理进程..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        echo "   [完成] 停止监控程序 (PID: $SERVER_PID)"
    fi
    if [ ! -z "$CLIENT_PID" ]; then
        kill $CLIENT_PID 2>/dev/null || true
        echo "   [完成] 停止目标程序 (PID: $CLIENT_PID)"
    fi
    sleep 1
}

trap cleanup EXIT INT TERM

echo "步骤 1: 启动 Bloom Filter 监控程序 (Server)"
echo "   作用: 加载 eBPF 程序，创建 bloom filter，监控用户访问"
echo "   命令: LD_PRELOAD=\"$SYSCALL_SERVER_LIB\" \"$UPROBE_PROGRAM\""
echo ""

# 启动监控程序
LD_PRELOAD="$SYSCALL_SERVER_LIB" "$UPROBE_PROGRAM" &
SERVER_PID=$!

echo "   [完成] 监控程序已启动 (PID: $SERVER_PID)"
echo "   [等待] 监控程序初始化..."
echo ""

# 等待 server 初始化
sleep 8

# 检查 server 是否正常运行
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[错误] 监控程序启动失败"
    wait $SERVER_PID 2>/dev/null || echo "监控程序退出码: $?"
    exit 1
fi

echo "步骤 2: 启动目标程序 (Client) 触发 Bloom Filter 测试"
echo "   作用: 调用 user_access() 函数，触发 bloom filter 检测"
echo "   命令: LD_PRELOAD=\"$AGENT_LIB\" \"$TARGET_PROGRAM\""
echo ""

echo "   [启动] 目标程序开始运行..."
LD_PRELOAD="$AGENT_LIB" timeout 20 "$TARGET_PROGRAM" &
CLIENT_PID=$!

echo "   [完成] 目标程序已启动 (PID: $CLIENT_PID)"
echo "   [进行中] 目标程序正在触发用户访问事件..."
echo ""

# 等待 client 完成
wait $CLIENT_PID 2>/dev/null || true
CLIENT_PID=""

echo "   [完成] 目标程序执行完成"
echo ""

echo "步骤 3: 观察 Bloom Filter 测试结果"
echo "   监控程序继续运行并分析 bloom filter 性能..."
echo ""

# 让 server 继续运行一段时间显示最终统计
sleep 15

echo ""
echo "测试完成！"
echo ""
echo "Bloom Filter 测试验证："
echo "  [测试方法] 与 queue_demo 类似的 Server/Client 架构"
echo "  [Server端] 监控程序使用 bloom filter 分析用户访问模式"
echo "  [Client端] 目标程序调用函数触发 uprobe 事件"
echo "  [验证点] 新用户检测、重复用户识别、假阳性率分析"
echo ""
echo "关键指标："
echo "  - 新用户数: 首次访问的用户（bloom filter 未命中）"
echo "  - 重复用户数: 再次访问的用户（bloom filter 命中）"
echo "  - 假阳性率: 误判为重复用户的新用户比例"
echo "  - 统计一致性: 新用户 + 重复用户 = 总访问次数"
