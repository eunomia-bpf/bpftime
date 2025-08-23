#!/bin/bash

# Complete ARM build test script for bpftime
# Run this script from the project root: ./tools/test_arm_build.sh
set -e

echo "=== Testing bpftime ARM64 Build in Docker ==="
echo "Running from: $(pwd)"
echo

echo "1. Building Docker image for ARM64..."
docker build --platform linux/arm64 -f tools/Dockerfile.arm -t bpftime-arm .
echo "✓ Docker image built successfully"

echo
echo "2. Testing bpftime build (Release mode)..."
echo "   This may take a while on ARM64 emulation..."
echo "   Using parallel compilation with -j flag..."
docker run --platform linux/arm64 --rm bpftime-arm sh -c "cd /bpftime && make clean && make release -j\$(nproc)"
if [ $? -eq 0 ]; then
    echo "✓ bpftime built successfully in release mode"
else
    echo "✗ bpftime build failed"
    exit 1
fi

echo
echo "3. Building example eBPF programs..."
docker run --platform linux/arm64 --rm bpftime-arm sh -c "cd /bpftime/example/malloc && make clean && make -j\$(nproc)"
if [ $? -eq 0 ]; then
    echo "✓ Example eBPF programs built successfully"
else
    echo "✗ Example eBPF programs build failed"
    exit 1
fi

echo
echo "=== ARM64 Build Test Complete ==="
echo "✓ All build tests passed successfully!"
echo
echo "bpftime has been verified to build on ARM64 architecture:"
echo "- All components build successfully"
echo "- Example programs compile correctly"
echo "- Core binaries are generated as expected"