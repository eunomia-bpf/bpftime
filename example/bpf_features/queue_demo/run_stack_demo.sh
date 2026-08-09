#!/bin/bash

# Stack Demo Dedicated Test Script
# Dedicated test for BPF Stack Map LIFO (Last In, First Out) functionality

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
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

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v clang &>/dev/null; then
        print_error "clang is not installed"
        exit 1
    fi

    if ! command -v llvm-strip &>/dev/null; then
        print_error "llvm-strip is not installed"
        exit 1
    fi

    # Check bpftime build files
    if [[ ! -f "../../build/runtime/syscall-server/libbpftime-syscall-server.so" ]]; then
        print_warning "bpftime server library file not found"
        print_info "Please run 'make build' in bpftime root directory first"
        exit 1
    fi

    if [[ ! -f "../../build/runtime/agent/libbpftime-agent.so" ]]; then
        print_warning "bpftime agent library file not found"
        print_info "Please run 'make build' in bpftime root directory first"
        exit 1
    fi

    print_success "Dependencies check completed"
}

# Build project
build_project() {
    print_info "Building project..."

    if ! make clean; then
        print_error "Clean failed"
        exit 1
    fi

    if ! make; then
        print_error "Build failed"
        exit 1
    fi

    print_success "Build completed"
}

# Check build status
check_build_status() {
    if [[ ! -x "./uprobe_stack" ]] || [[ ! -x "./target" ]]; then
        print_warning "Programs not built or incomplete"
        print_info "Building now..."
        build_project
    fi
}

# Run Stack demo - server-client mode
run_stack_demo_server_client() {
    print_info "=== BPF Stack Map Demo (LIFO - Last In, First Out) ==="
    print_info "Starting Stack Demo (Server-Client mode)..."
    echo

    # Check if programs exist
    if [[ ! -x "./uprobe_stack" ]]; then
        print_error "uprobe_stack program does not exist, please build first"
        exit 1
    fi

    if [[ ! -x "./target" ]]; then
        print_error "target program does not exist, please build first"
        exit 1
    fi

    # Start server side
    print_info "Starting Stack Server (monitor program)..."
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack &
    SERVER_PID=$!

    # Wait for server to start
    sleep 3
    print_success "Stack Server started (PID: $SERVER_PID)"

    # Start client side
    print_info "Starting Client (target program)..."
    LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target &
    CLIENT_PID=$!

    # Wait for client to start
    sleep 2
    print_success "Client started (PID: $CLIENT_PID)"

    # Setup cleanup function on exit
    cleanup() {
        print_info "Cleaning up processes..."

        if kill -0 $CLIENT_PID 2>/dev/null; then
            kill $CLIENT_PID
            print_info "Client stopped"
        fi

        if kill -0 $SERVER_PID 2>/dev/null; then
            kill $SERVER_PID
            print_info "Stack Server stopped"
        fi

        exit 0
    }

    trap cleanup SIGINT SIGTERM

    print_info "Stack Demo is running..."
    print_info "Note: Stack is LIFO (Last In, First Out), events are processed in reverse chronological order"
    print_info "Press Ctrl+C to stop demo"
    print_info "Or run 'pkill -f uprobe_stack' and 'pkill -f target' in another terminal to stop"
    echo

    # Wait for user interruption
    while true; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            print_error "Stack Server exited unexpectedly"
            break
        fi

        if ! kill -0 $CLIENT_PID 2>/dev/null; then
            print_error "Client exited unexpectedly"
            break
        fi

        sleep 5
    done

    # Cleanup
    cleanup
}

# Run Stack demo - manual mode (provide guidance)
run_stack_demo_manual() {
    print_info "Stack Demo Manual Mode - Please follow these steps:"
    echo
    print_info "Terminal 1 - Start Stack Server (monitor program):"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack"
    echo
    print_info "Terminal 2 - Start Client (target program):"
    echo "  cd $(pwd)"
    echo "  LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target"
    echo
    print_warning "Note: Please run Server first, then Client"
    print_info "Stack is LIFO (Last In, First Out), events are processed in reverse chronological order"
}

# Demonstrate bpftime command line tool usage
run_stack_demo_bpftime_cli() {
    print_info "Stack Demo bpftime CLI Mode - Please follow these steps:"
    echo
    print_info "Step 1 - Load eBPF program in bpftime root directory:"
    echo "  cd ../../"
    echo "  sudo bpftime load example/queue_demo/uprobe_stack"
    echo
    print_info "Step 2 - Run target program in another terminal:"
    echo "  cd example/queue_demo/"
    echo "  sudo bpftime start ./target"
    echo
    print_warning "Note: This mode requires sudo privileges"
    print_info "Stack is LIFO (Last In, First Out), events are processed in reverse chronological order"
}

# Show usage
show_usage() {
    echo "Stack Map Demo Dedicated Test Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show help information"
    echo "  -b, --build        Build project only, do not run"
    echo "  -r, --run          Run Stack demo (server-client auto mode)"
    echo "  -m, --manual       Show manual running guidance"
    echo "  -t, --bpftime-cli  Show bpftime command line tool usage"
    echo "  -c, --clean        Clean build files"
    echo ""
    echo "Default behavior: Build and run Stack demo (server-client auto mode)"
    echo ""
    echo "Stack characteristics:"
    echo "  - LIFO (Last In, First Out)"
    echo "  - Events are processed in reverse chronological order"
    echo "  - Latest events are popped first"
    echo ""
    echo "Other scripts:"
    echo "  ./run_demo.sh        - Comprehensive test for Queue and Stack functionality"
    echo "  ./run_queue_demo.sh  - Dedicated Queue functionality test"
}

# Clean function
clean_project() {
    print_info "Cleaning project..."
    make clean
    print_success "Clean completed"
}

# Main function
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
        run_stack_demo_server_client
        exit 0
        ;;
    -m | --manual)
        check_dependencies
        check_build_status
        run_stack_demo_manual
        exit 0
        ;;
    -t | --bpftime-cli)
        check_dependencies
        check_build_status
        run_stack_demo_bpftime_cli
        exit 0
        ;;
    -c | --clean)
        clean_project
        exit 0
        ;;
    "")
        # Default: build and run Stack demo
        check_dependencies
        build_project
        run_stack_demo_server_client
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
}

# Execute main function
main "$@"
