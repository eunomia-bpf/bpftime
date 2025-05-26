#!/bin/bash

# Queue Demo 运行脚本

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

    if ! command -v clang &>/dev/null; then
        print_error "clang 未安装"
        exit 1
    fi

    if ! command -v llvm-strip &>/dev/null; then
        print_error "llvm-strip 未安装"
        exit 1
    fi

    # 检查bpftime构建文件
    if [[ ! -f "../../build/runtime/syscall-server/libbpftime-syscall-server.so" ]]; then
        print_warning "未找到bpftime server库文件"
        print_info "请先在bpftime根目录运行 'make build'"
    fi

    if [[ ! -f "../../build/runtime/agent/libbpftime-agent.so" ]]; then
        print_warning "未找到bpftime agent库文件"
        print_info "请先在bpftime根目录运行 'make build'"
    fi

    print_success "依赖检查完成"
}

# 构建项目
build_project() {
    print_info "构建项目..."

    if ! make clean; then
        print_error "清理失败"
        exit 1
    fi

    if ! make; then
        print_error "构建失败"
        exit 1
    fi

    print_success "构建完成"
}

# 运行demo - server-client模式
run_demo_server_client() {
    print_info "开始运行Demo (Server-Client模式)..."

    # 检查程序是否存在
    if [[ ! -x "./uprobe_queue" ]]; then
        print_error "uprobe_queue 程序不存在，请先构建"
        exit 1
    fi

    if [[ ! -x "./target" ]]; then
        print_error "target 程序不存在，请先构建"
        exit 1
    fi

    # 启动server端
    print_info "启动Server端 (监控程序)..."
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue &
    SERVER_PID=$!

    # 等待server启动
    sleep 3
    print_success "Server端已启动 (PID: $SERVER_PID)"

    # 启动client端
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
            print_info "Server端已停止"
        fi

        exit 0
    }

    trap cleanup SIGINT SIGTERM

    print_info "Demo正在运行..."
    print_info "按 Ctrl+C 停止demo"
    print_info "或者在另一个终端运行 'pkill -f uprobe_queue' 和 'pkill -f target' 来停止"
    echo

    # 等待用户中断
    while true; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            print_error "Server端意外退出"
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

# 运行demo - 手动模式（给出指导）
run_demo_manual() {
    print_info "手动运行模式 - 请按以下步骤操作:"
    echo
    print_info "终端1 - 启动Server端 (监控程序):"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue"
    echo
    print_info "终端2 - 启动Client端 (目标程序):"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target"
    echo
    print_warning "注意: 请先运行Server端，再运行Client端"
}

# 演示bpftime命令行工具用法
run_demo_bpftime_cli() {
    print_info "bpftime命令行工具模式 - 请按以下步骤操作:"
    echo
    print_info "步骤1 - 在bpftime根目录加载eBPF程序:"
    echo "  cd ../../"
    echo "  sudo bpftime load example/queue_demo/uprobe_queue"
    echo
    print_info "步骤2 - 在另一个终端运行目标程序:"
    echo "  cd example/queue_demo/"
    echo "  sudo bpftime start ./target"
    echo
    print_warning "注意: 此模式需要sudo权限"
}

# 显示使用方法
show_usage() {
    echo "Queue Map Demo 运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help         显示帮助信息"
    echo "  -b, --build        只构建项目，不运行"
    echo "  -r, --run          运行demo (server-client自动模式)"
    echo "  -m, --manual       显示手动运行指导"
    echo "  -t, --bpftime-cli  显示bpftime命令行工具用法"
    echo "  -c, --clean        清理构建文件"
    echo ""
    echo "默认行为: 构建并运行demo (server-client自动模式)"
    echo ""
    echo "运行模式说明:"
    echo "  1. Server-Client自动模式: 脚本自动启动server和client"
    echo "  2. 手动模式: 提供命令指导，用户手动在两个终端运行"
    echo "  3. bpftime CLI模式: 使用bpftime命令行工具"
}

# 清理函数
clean_project() {
    print_info "清理项目..."
    make clean
    print_success "清理完成"
}

# 检查构建状态
check_build_status() {
    if [[ ! -x "./uprobe_queue" ]] || [[ ! -x "./target" ]]; then
        print_warning "程序未构建或构建不完整"
        print_info "正在构建..."
        build_project
    fi
}

# 主函数
main() {
    case ${1:-""} in
    -h | --help)
        show_usage
        exit 0
        ;;
    -b | --build)
        check_dependencies
        build_project
        exit 0
        ;;
    -r | --run)
        check_dependencies
        check_build_status
        run_demo_server_client
        exit 0
        ;;
    -m | --manual)
        check_dependencies
        check_build_status
        run_demo_manual
        exit 0
        ;;
    -t | --bpftime-cli)
        check_dependencies
        check_build_status
        run_demo_bpftime_cli
        exit 0
        ;;
    -c | --clean)
        clean_project
        exit 0
        ;;
    "")
        # 默认：构建并运行
        check_dependencies
        build_project
        run_demo_server_client
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
