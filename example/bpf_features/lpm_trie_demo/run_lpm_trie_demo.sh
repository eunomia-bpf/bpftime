#!/bin/bash
# SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
# bpftime LPM Trie demo script

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_step "Checking dependencies..."

    # Check bpftime build
    if [ ! -f "../../build/runtime/syscall-server/libbpftime-syscall-server.so" ]; then
        print_error "bpftime syscall-server not found, please build bpftime first"
        print_info "Run: cd ../../ && make build"
        exit 1
    fi

    if [ ! -f "../../build/runtime/agent/libbpftime-agent.so" ]; then
        print_error "bpftime agent not found, please build bpftime first"
        print_info "Run: cd ../../ && make build"
        exit 1
    fi

    print_success "bpftime dependencies check passed"
}

# Build programs
build_programs() {
    print_step "Building LPM Trie demo programs..."

    # Clean old files
    make clean >/dev/null 2>&1 || true

    # Build BPF program and userspace programs
    if ! make all; then
        print_error "Build failed"
        exit 1
    fi

    print_success "Build completed"
}

# Prepare test environment
prepare_test_env() {
    print_step "Preparing test environment..."

    # Create test directories and files
    mkdir -p /tmp/test_allowed
    echo "test content" >/tmp/test_allowed/file1.txt
    echo "test content" >/tmp/test_allowed/file2.txt

    # Ensure target programs are executable
    chmod +x ./file_access_target
    chmod +x ./file_access_monitor

    print_success "Test environment prepared"
}

# Start monitor program
start_monitor() {
    print_step "Starting file access monitor..."

    # Get absolute path of current directory
    CURRENT_DIR=$(pwd)
    TARGET_PATH="$CURRENT_DIR/file_access_target"

    print_info "Monitor target: $TARGET_PATH"
    print_info "Using bpftime syscall-server mode"

    # Start monitor program (as bpftime server)
    LD_PRELOAD=../../build/runtime/syscall-server/libbpftime-syscall-server.so \
        ./file_access_monitor "$TARGET_PATH" &

    MONITOR_PID=$!
    echo $MONITOR_PID >/tmp/lpm_trie_monitor.pid

    print_success "Monitor started (PID: $MONITOR_PID)"

    # Wait for monitor initialization
    sleep 2
}

# Run test program
run_tests() {
    print_step "Running file access tests..."

    print_info "Using bpftime agent mode to run test program"

    # Run test program (as bpftime client)
    LD_PRELOAD=../../build/runtime/agent/libbpftime-agent.so \
        ./file_access_target

    print_success "Test program execution completed"
}

# Stop monitor program
stop_monitor() {
    print_step "Stopping monitor..."

    if [ -f /tmp/lpm_trie_monitor.pid ]; then
        MONITOR_PID=$(cat /tmp/lpm_trie_monitor.pid)
        if kill -0 $MONITOR_PID 2>/dev/null; then
            kill -SIGINT $MONITOR_PID
            sleep 1

            # Force kill if still running
            if kill -0 $MONITOR_PID 2>/dev/null; then
                kill -9 $MONITOR_PID
            fi
        fi
        rm -f /tmp/lpm_trie_monitor.pid
    fi

    print_success "Monitor stopped"
}

# Cleanup function
cleanup() {
    print_step "Cleaning up resources..."
    stop_monitor
    rm -f /tmp/test_allowed/file1.txt /tmp/test_allowed/file2.txt
    rmdir /tmp/test_allowed 2>/dev/null || true
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "bpftime LPM Trie demo script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --clean    Clean build files only"
    echo "  -b, --build    Build programs only"
    echo "  -t, --test     Run tests only"
    echo ""
    echo "Default behavior: Full demo (build + run tests)"
}

# Main function
main() {
    echo -e "${CYAN}"
    echo "========================================"
    echo "  bpftime LPM Trie Demo"
    echo "========================================"
    echo -e "${NC}"

    # Parse command line arguments
    case "${1:-}" in
    -h | --help)
        show_help
        exit 0
        ;;
    -c | --clean)
        make clean
        print_success "Cleanup completed"
        exit 0
        ;;
    -b | --build)
        check_dependencies
        build_programs
        exit 0
        ;;
    -t | --test)
        prepare_test_env
        start_monitor
        sleep 1
        run_tests
        sleep 2
        stop_monitor
        exit 0
        ;;
    "")
        # Default full demo
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac

    # Set cleanup trap
    trap cleanup EXIT INT TERM

    # Execute full demo
    check_dependencies
    build_programs
    prepare_test_env
    start_monitor

    print_info "Waiting for monitor to fully start..."
    sleep 3

    run_tests

    print_info "Waiting for event processing to complete..."
    sleep 3

    stop_monitor

    echo ""
    echo -e "${GREEN}========================================"
    echo "  LPM Trie Demo Completed!"
    echo "========================================"
    echo -e "${NC}"

    print_info "Demo highlights:"
    echo "  - Used real BPF_MAP_TYPE_LPM_TRIE"
    echo "  - Implemented complete bpftime client/server architecture"
    echo "  - Demonstrated longest prefix matching functionality"
    echo "  - Implemented real-time file access monitoring"

    print_info "To run demo again, execute: $0"
}

# Run main function
main "$@"
