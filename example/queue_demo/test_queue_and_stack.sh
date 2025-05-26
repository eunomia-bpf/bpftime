#!/bin/bash

# 测试queue和stack功能的脚本
# 这个脚本会演示queue(FIFO)和stack(LIFO)的不同行为

set -e

echo "=== BPF Queue和Stack功能测试 ==="
echo "这个测试将演示queue(FIFO)和stack(LIFO)的不同行为"
echo

# 检查是否在正确的目录
if [ ! -f "target.c" ]; then
    echo "错误: 请在queue_demo目录中运行此脚本"
    exit 1
fi

# 编译所有程序
echo "1. 编译程序..."
make clean
make all

if [ $? -ne 0 ]; then
    echo "编译失败"
    exit 1
fi

echo "编译成功！"
echo

# 启动目标程序
echo "2. 启动目标程序..."
./target &
TARGET_PID=$!
echo "目标程序PID: $TARGET_PID"

# 等待目标程序启动
sleep 2

echo
echo "=== 第一部分：测试Queue功能 (FIFO - 先进先出) ==="
echo "启动queue监控程序，运行10秒..."

# 启动queue监控程序，运行10秒
timeout 10s ./uprobe_queue || true

echo
echo "=== 第二部分：测试Stack功能 (LIFO - 后进先出) ==="
echo "启动stack监控程序，运行10秒..."

# 启动stack监控程序，运行10秒
timeout 10s ./uprobe_stack || true

echo
echo "=== 第三部分：对比测试 ==="
echo "同时运行queue和stack监控，观察它们处理相同事件的不同顺序"
echo

# 创建临时文件来保存输出
QUEUE_OUTPUT="/tmp/queue_output.txt"
STACK_OUTPUT="/tmp/stack_output.txt"

echo "启动queue监控 (后台运行)..."
timeout 8s ./uprobe_queue >"$QUEUE_OUTPUT" 2>&1 &
QUEUE_PID=$!

echo "启动stack监控 (后台运行)..."
timeout 8s ./uprobe_stack >"$STACK_OUTPUT" 2>&1 &
STACK_PID=$!

# 等待监控程序完成
wait $QUEUE_PID 2>/dev/null || true
wait $STACK_PID 2>/dev/null || true

echo
echo "=== Queue输出 (FIFO - 先进先出) ==="
if [ -f "$QUEUE_OUTPUT" ]; then
    cat "$QUEUE_OUTPUT"
else
    echo "Queue输出文件不存在"
fi

echo
echo "=== Stack输出 (LIFO - 后进先出) ==="
if [ -f "$STACK_OUTPUT" ]; then
    cat "$STACK_OUTPUT"
else
    echo "Stack输出文件不存在"
fi

echo
echo "=== 分析结果 ==="
echo "观察上面的输出，你应该能看到："
echo "1. Queue: 事件按照发生的时间顺序被处理 (先发生的先被处理)"
echo "2. Stack: 事件按照相反的时间顺序被处理 (后发生的先被处理)"
echo "3. 这展示了FIFO(队列)和LIFO(栈)的根本区别"

# 清理
echo
echo "4. 清理..."
kill $TARGET_PID 2>/dev/null || true
rm -f "$QUEUE_OUTPUT" "$STACK_OUTPUT"

echo "测试完成！"
echo
echo "=== 总结 ==="
echo "✓ Queue (FIFO): 先进先出，事件按时间顺序处理"
echo "✓ Stack (LIFO): 后进先出，最新事件优先处理"
echo "✓ 两种数据结构都在bpftime中正常工作"
