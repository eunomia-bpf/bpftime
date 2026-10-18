#!/bin/bash

# Correct Bloom Filter test script
# Similar to queue_demo test approach: Server monitors, Client triggers events
set -e

# VM Configuration:
# - Default VM (llvm): Works well, better performance
# - ubpf VM: May have agent compatibility issues in some builds
# - Choose based on your bpftime build configuration
# export BPFTIME_VM_NAME=ubpf  # Uncomment if ubpf is available
export BPFTIME_VM_NAME=llvm # Explicitly use LLVM VM (recommended)

echo "=========================================="
echo "  Correct Bloom Filter Test Demo"
echo "=========================================="
echo ""

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check files
if [ ! -f "./uprobe_bloom_filter" ] || [ ! -f "./target" ]; then
    echo "[Error] Please run 'make' to build programs first"
    exit 1
fi

# Use absolute paths
UPROBE_PROGRAM="$SCRIPT_DIR/uprobe_bloom_filter"
TARGET_PROGRAM="$SCRIPT_DIR/target"
SYSCALL_SERVER_LIB="$SCRIPT_DIR/../../build/runtime/syscall-server/libbpftime-syscall-server.so"
AGENT_LIB="$SCRIPT_DIR/../../build/runtime/agent/libbpftime-agent.so"

if [ ! -f "$SYSCALL_SERVER_LIB" ] || [ ! -f "$AGENT_LIB" ]; then
    echo "[Error] bpftime library files not found, please build bpftime first"
    echo "  Expected location: $SYSCALL_SERVER_LIB"
    echo "  Expected location: $AGENT_LIB"
    exit 1
fi

echo "Test principle explanation:"
echo "  Similar to queue_demo test approach:"
echo "  1. Server: Monitor program runs, waiting for uprobe events"
echo "  2. Client: Target program runs, triggering function calls"
echo "  3. Verification: Server analyzes user access patterns through bloom filter"
echo "  Using VM: ${BPFTIME_VM_NAME:-default (llvm)}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        echo "   [Done] Stopped monitor program (PID: $SERVER_PID)"
    fi
    if [ ! -z "$CLIENT_PID" ]; then
        kill $CLIENT_PID 2>/dev/null || true
        echo "   [Done] Stopped target program (PID: $CLIENT_PID)"
    fi
    sleep 1
}

trap cleanup EXIT INT TERM

echo "Step 1: Start Bloom Filter Monitor Program (Server)"
echo "   Purpose: Load eBPF program, create bloom filter, monitor user access"
echo "   Command: LD_PRELOAD=\"$SYSCALL_SERVER_LIB\" \"$UPROBE_PROGRAM\""
echo ""

# Start monitor program
LD_PRELOAD="$SYSCALL_SERVER_LIB" "$UPROBE_PROGRAM" &
SERVER_PID=$!

echo "   [Done] Monitor program started (PID: $SERVER_PID)"
echo "   [Waiting] Monitor program initializing..."
echo ""

# Wait for server initialization
sleep 8

# Check if server is running properly
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[Error] Monitor program failed to start"
    wait $SERVER_PID 2>/dev/null || echo "Monitor program exit code: $?"
    exit 1
fi

echo "Step 2: Start Target Program (Client) to Trigger Bloom Filter Test"
echo "   Purpose: Call user_access() function to trigger bloom filter detection"
echo "   Command: LD_PRELOAD=\"$AGENT_LIB\" \"$TARGET_PROGRAM\""
echo ""

echo "   [Starting] Target program begins execution..."
LD_PRELOAD="$AGENT_LIB" timeout 20 "$TARGET_PROGRAM" &
CLIENT_PID=$!

echo "   [Done] Target program started (PID: $CLIENT_PID)"
echo "   [In Progress] Target program triggering user access events..."
echo ""

# Wait for client completion
wait $CLIENT_PID 2>/dev/null || true
CLIENT_PID=""

echo "   [Done] Target program execution completed"
echo ""

echo "Step 3: Observe Bloom Filter Test Results"
echo "   Monitor program continues running and analyzing bloom filter performance..."
echo ""

# Let server continue running for a while to show final statistics
sleep 15

echo ""
echo "Test completed!"
echo ""
echo "Bloom Filter Test Verification:"
echo "  [Test Method] Server/Client architecture similar to queue_demo"
echo "  [Server Side] Monitor program uses bloom filter to analyze user access patterns"
echo "  [Client Side] Target program calls functions to trigger uprobe events"
echo "  [Verification Points] New user detection, repeat user identification, false positive rate analysis"
echo ""
echo "Key Metrics:"
echo "  - New users: First-time access users (bloom filter miss)"
echo "  - Repeat users: Re-access users (bloom filter hit)"
echo "  - False positive rate: New users misjudged as repeat users ratio"
echo "  - Statistical consistency: new users + repeat users = total accesses"
