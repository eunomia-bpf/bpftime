#!/bin/bash

# This script builds the dynamic load module, then builds the filter implementation library,
# and finally runs a test to verify it works with environment variables.

set -e  # Exit on any error

# Build directory
BUILD_DIR="/home/yunwei37/bpftime/build"
ATTACH_DIR="/home/yunwei37/bpftime/example/attach_implementation"
DYNAMIC_DIR="$ATTACH_DIR/benchmark/dynamic_load_plugin"
LIBS_DIR="$DYNAMIC_DIR/libs"

# Step 1: Build the module
echo "Building the nginx module..."
cd "$BUILD_DIR"
make nginx_build

# Step 2: Build the filter implementation
echo "Building the filter implementation library..."
cd "$LIBS_DIR"
make clean && make

# Path to the built library
LIB_PATH="$LIBS_DIR/libfilter_impl.so"

# Step 3: Test the module with environment variables
echo "Testing the module with environment variables..."
cd "$ATTACH_DIR"
export DYNAMIC_LOAD_LIB_PATH="$LIB_PATH"
export DYNAMIC_LOAD_URL_PREFIX="/aaaa"

echo "Environment variables set:"
echo "DYNAMIC_LOAD_LIB_PATH=$DYNAMIC_LOAD_LIB_PATH"
echo "DYNAMIC_LOAD_URL_PREFIX=$DYNAMIC_LOAD_URL_PREFIX"

echo "Running nginx configuration test..."
./nginx_plugin_output/nginx -p "$ATTACH_DIR" -c benchmark/dynamic_load_module.conf -t

echo "Done!" 