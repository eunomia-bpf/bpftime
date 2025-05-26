#!/bin/bash

# Stack Demo 运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."

    # 检查bpftime构建文件
    if [[ ! -f "../../build/runtime/syscall-server/libbpftime-syscall-server.so" ]]; then
        print_warning "未找到bpftime server库文件"
        print_info "请先在bpftime根目录运行 'make build'"
        exit 1
    fi

    if [[ ! -f "../../build/runtime/agent/libbpftime-agent.so" ]]; then
        print_warning "未找到bpftime agent库文件"
        print_info "请先在bpftime根目录运行 'make build'"
        exit 1
    fi

    print_success "依赖检查完成"
}

# 运行stack demo
run_stack_demo() {
    print_info "开始运行Stack Demo (LIFO - 后进先出)..."

    # 检查程序是否存在
    if [[ ! -x "./uprobe_stack" ]]; then
        print_error "uprobe_stack 程序不存在，请先构建"
        exit 1
    fi

    if [[ ! -x "./target" ]]; then
        print_error "target 程序不存在，请先构建"
        exit 1
    fi

    # 启动server端 (stack监控程序)
    print_info "启动Stack Server端 (监控程序)..."
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack &
    SERVER_PID=$!

    # 等待server启动
    sleep 3
    print_success "Stack Server端已启动 (PID: $SERVER_PID)"

    # 启动client端 (目标程序)
    print_info "启动Client端 (目标程序)..."
    LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target &
    CLIENT_PID=$!

    # 等待client启动
    sleep 2
    print_success "Client端已启动 (PID: $CLIENT_PID)"

    # 设置退出时清理函数
    cleanup() {
        print_info "清理进程..."

        if kill -0 $CLIENT_PID 2>/dev/null; then
            kill $CLIENT_PID
            print_info "Client端已停止"
        fi

        if kill -0 $SERVER_PID 2>/dev/null; then
            kill $SERVER_PID
            print_info "Stack Server端已停止"
        fi

        exit 0
    }

    trap cleanup SIGINT SIGTERM

    print_info "Stack Demo正在运行..."
    print_info "注意：栈是LIFO(后进先出)，最新的事件会最先被弹出"
    print_info "按 Ctrl+C 停止demo"
    echo

    # 等待用户中断
    while true; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            print_error "Stack Server端意外退出"
            break
        fi

        if ! kill -0 $CLIENT_PID 2>/dev/null; then
            print_error "Client端意外退出"
            break
        fi

        sleep 5
    done

    # 清理
    cleanup
}

# 对比测试 - 同时运行queue和stack
run_comparison_demo() {
    print_info "开始运行对比测试 (Queue vs Stack)..."

    # 检查程序是否存在
    if [[ ! -x "./uprobe_queue" ]] || [[ ! -x "./uprobe_stack" ]] || [[ ! -x "./target" ]]; then
        print_error "程序不完整，请先构建所有程序"
        exit 1
    fi

    print_info "这个测试将同时运行queue和stack监控，展示FIFO和LIFO的区别"
    echo

    # 启动queue server
    print_info "启动Queue Server端..."
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue >/tmp/queue_output.log 2>&1 &
    QUEUE_PID=$!
    sleep 2

    # 启动stack server
    print_info "启动Stack Server端..."
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack >/tmp/stack_output.log 2>&1 &
    STACK_PID=$!
    sleep 2

    # 启动目标程序
    print_info "启动目标程序..."
    LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target &
    TARGET_PID=$!
    sleep 2

    print_success "所有程序已启动"
    print_info "运行10秒后自动停止..."

    # 运行10秒
    sleep 10

    # 停止所有程序
    print_info "停止所有程序..."
    kill $TARGET_PID 2>/dev/null || true
    sleep 1
    kill $QUEUE_PID 2>/dev/null || true
    kill $STACK_PID 2>/dev/null || true
    sleep 1

    # 显示结果
    echo
    print_info "=== Queue输出 (FIFO - 先进先出) ==="
    if [[ -f "/tmp/queue_output.log" ]]; then
        cat /tmp/queue_output.log
    else
        print_warning "Queue输出文件不存在"
    fi

    echo
    print_info "=== Stack输出 (LIFO - 后进先出) ==="
    if [[ -f "/tmp/stack_output.log" ]]; then
        cat /tmp/stack_output.log
    else
        print_warning "Stack输出文件不存在"
    fi

    # 清理临时文件
    rm -f /tmp/queue_output.log /tmp/stack_output.log

    echo
    print_success "对比测试完成！"
    print_info "观察上面的输出，你应该能看到Queue和Stack处理事件的不同顺序"
}

# 显示使用方法
show_usage() {
    echo "Stack Demo 运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help         显示帮助信息"
    echo "  -s, --stack        运行stack demo"
    echo "  -c, --compare      运行对比测试 (queue vs stack)"
    echo "  -m, --manual       显示手动运行指导"
    echo ""
    echo "默认行为: 运行stack demo"
}

# 手动运行指导
show_manual() {
    print_info "手动运行Stack Demo - 请按以下步骤操作:"
    echo
    print_info "终端1 - 启动Stack Server端:"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack"
    echo
    print_info "终端2 - 启动Client端:"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target"
    echo
    print_warning "注意: 请先运行Server端，再运行Client端"
    echo
    print_info "对比测试 - 在不同终端同时运行:"
    echo "  终端1: LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue"
    echo "  终端2: LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack"
    echo "  终端3: LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target"
}

# 主函数
main() {
    case ${1:-""} in
    -h | --help)
        show_usage
        exit 0
        ;;
    -s | --stack)
        check_dependencies
        run_stack_demo
        exit 0
        ;;
    -c | --compare)
        check_dependencies
        run_comparison_demo
        exit 0
        ;;
    -m | --manual)
        show_manual
        exit 0
        ;;
    "")
        # 默认：运行stack demo
        check_dependencies
        run_stack_demo
        ;;
    *)
        print_error "未知选项: $1"
        show_usage
        exit 1
        ;;
    esac
}

# 执行主函数
main "$@"
