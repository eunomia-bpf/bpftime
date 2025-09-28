#!/bin/bash

# Queue and Stack Comprehensive Test Script
# This script demonstrates the different behaviors of queue(FIFO) and stack(LIFO)

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
    if [[ ! -x "./uprobe_queue" ]] || [[ ! -x "./uprobe_stack" ]] || [[ ! -x "./target" ]]; then
        print_warning "Programs not built or incomplete"
        print_info "Building now..."
        build_project
    fi
}

# Run comprehensive test
run_comprehensive_test() {
    print_info "=== BPF Queue and Stack Comprehensive Test ==="
    print_info "This test demonstrates the different behaviors of queue(FIFO) and stack(LIFO)"
    echo

    # Check if programs exist
    if [[ ! -x "./uprobe_queue" ]] || [[ ! -x "./uprobe_stack" ]] || [[ ! -x "./target" ]]; then
        print_error "Programs incomplete, please build all programs first"
        exit 1
    fi

    # Start target program
    print_info "Starting target program..."
    LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so ./target &
    TARGET_PID=$!
    print_success "Target program started (PID: $TARGET_PID)"

    # Wait for target program to start
    sleep 3

    # Setup cleanup function
    cleanup() {
        print_info "Cleaning up processes..."
        kill $TARGET_PID 2>/dev/null || true
        rm -f /tmp/queue_output.log /tmp/stack_output.log
        exit 0
    }
    trap cleanup SIGINT SIGTERM

    echo
    print_info "=== Part 1: Testing Queue functionality (FIFO - First In, First Out) ==="
    print_info "Starting queue monitor program, running for 10 seconds..."

    # Start queue monitor program for 10 seconds
    timeout 10s bash -c "LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue" || true

    echo
    print_info "=== Part 2: Testing Stack functionality (LIFO - Last In, First Out) ==="
    print_info "Starting stack monitor program, running for 10 seconds..."

    # Start stack monitor program for 10 seconds
    timeout 10s bash -c "LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack" || true

    echo
    print_info "=== Part 3: Comparison Test ==="
    print_info "Running queue and stack monitors simultaneously to observe different event processing orders"
    echo

    # Create temporary files to save output
    QUEUE_OUTPUT="/tmp/queue_output.log"
    STACK_OUTPUT="/tmp/stack_output.log"

    print_info "Starting queue monitor (background)..."
    timeout 8s bash -c "LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_queue" >"$QUEUE_OUTPUT" 2>&1 &
    QUEUE_PID=$!

    print_info "Starting stack monitor (background)..."
    timeout 8s bash -c "LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so ./uprobe_stack" >"$STACK_OUTPUT" 2>&1 &
    STACK_PID=$!

    # Wait for monitor programs to complete
    wait $QUEUE_PID 2>/dev/null || true
    wait $STACK_PID 2>/dev/null || true

    echo
    print_info "=== Queue Output (FIFO - First In, First Out) ==="
    if [ -f "$QUEUE_OUTPUT" ]; then
        cat "$QUEUE_OUTPUT"
    else
        print_warning "Queue output file does not exist"
    fi

    echo
    print_info "=== Stack Output (LIFO - Last In, First Out) ==="
    if [ -f "$STACK_OUTPUT" ]; then
        cat "$STACK_OUTPUT"
    else
        print_warning "Stack output file does not exist"
    fi

    echo
    print_info "=== Analysis Results ==="
    print_success "From the output above, you should observe:"
    print_success "1. Queue: Events are processed in chronological order (first occurred, first processed)"
    print_success "2. Stack: Events are processed in reverse chronological order (last occurred, first processed)"
    print_success "3. This demonstrates the fundamental difference between FIFO(queue) and LIFO(stack)"

    # Cleanup
    cleanup
}

# Show usage
show_usage() {
    echo "Queue and Stack Comprehensive Test Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show help information"
    echo "  -b, --build        Build project only, do not run"
    echo "  -r, --run          Run comprehensive test"
    echo "  -c, --clean        Clean build files"
    echo ""
    echo "Default behavior: Build and run comprehensive test"
    echo ""
    echo "Test content:"
    echo "  1. Queue functionality test (FIFO - First In, First Out)"
    echo "  2. Stack functionality test (LIFO - Last In, First Out)"
    echo "  3. Comparison test (run both monitors simultaneously)"
    echo ""
    echo "Other scripts:"
    echo "  ./run_queue_demo.sh  - Dedicated Queue functionality test"
    echo "  ./run_stack_demo.sh  - Dedicated Stack functionality test"
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
        run_comprehensive_test
        exit 0
        ;;
    -c | --clean)
        clean_project
        exit 0
        ;;
    "")
        # Default: build and run comprehensive test
        check_dependencies
        build_project
        run_comprehensive_test
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
