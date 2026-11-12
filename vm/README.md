# bpftime Virtual Machine (VM) Architecture

The bpftime VM subsystem provides a high-performance, modular framework for executing eBPF programs in userspace. It features multiple execution backends, advanced optimization capabilities, and seamless integration with the bpftime runtime system.

## Architecture Overview

The VM subsystem employs a layered architecture with clean abstractions:

```
┌─────────────────────────────────────────────────────────────┐
│                    External Consumers                        │
├─────────────────────────────────────────────────────────────┤
│                 VM Core API (C Interface)                    │
│                    (vm-core/ebpf-vm.h)                      │
├─────────────────────────────────────────────────────────────┤
│              Compatibility Layer (C++ Interface)             │
│                 (compat/bpftime_vm_compat.hpp)              │
├─────────────┬─────────────────┬─────────────────────────────┤
│  LLVM JIT   │   uBPF Backend  │   Future Backends...        │
│  Backend    │                 │                              │
└─────────────┴─────────────────┴─────────────────────────────┘
```
See [example/main.cpp](example/main.cpp) for how to use it.

## Build

Build the vm only:

```sh
make build-llvm # build llvm backend
make build-ubpf # build ubpf backend
```

see [llvm-jit/README.md](llvm-jit/README.md) for how to use the LLVM JIT backend.

You can also build the llvm JIT/AOT for eBPF as a standalone library in it's own directory:

```sh
sudo apt install llvm-15-dev
cd llvm-jit
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -j
```

You can see the cli for how to use [AOT compile](cli/README.md).

see [ubpf-vm/README.md](ubpf-vm/README.md)

## cli for VM only.

A tool for loading and running eBPF programs in the VM only.

```console
$ bpftime-cli
Usage: build/vm/cli/bpftime-cli <path to ebpf instructions> [path to memory for the ebpf program]
```

See [cli](cli/README.md) for more details. Since cli is dependent on libbpf for loading eBPF programs, you need to compile it from the project root:

```sh
make release-with-llvm-jit
```

See [.github/workflows/test-aot-cli.yml](../.github/workflows/test-aot-cli.yml) for more details.

## Components

### 1. VM Core (`vm-core/`)

The VM core provides the primary C API interface for external consumers. It acts as a thin wrapper that delegates to the compatibility layer while maintaining a stable C ABI.

**Key Files:**
- `include/ebpf-vm.h` - Main C API header
- `src/ebpf-vm.cpp` - API implementation

**Core Functions:**
```c
// VM lifecycle management
struct ebpf_vm *ebpf_create(const char *vm_name);
void ebpf_destroy(struct ebpf_vm *vm);

// Code loading and execution
int ebpf_load(struct ebpf_vm *vm, const void *code, uint32_t code_len, char **errmsg);
int64_t ebpf_exec(struct ebpf_vm *vm, void *mem, size_t mem_len, int64_t *result);

// Compilation
ebpf_jit_fn ebpf_compile(struct ebpf_vm *vm, char **errmsg);
```

### 2. Compatibility Layer (`compat/`)

The compatibility layer provides a unified C++ abstraction for different VM backends. It implements a factory pattern for dynamic backend registration and selection.

**Key Classes:**
- `bpftime_vm_impl` - Abstract base class for VM implementations
- Factory registry for backend registration

**Key Features:**
- Pluggable VM backends
- Unified error handling
- Helper function management
- LDDW (Load Double Word) instruction support

### 3. LLVM JIT Backend (`llvm-jit/`)

High-performance JIT/AOT compiler using LLVM infrastructure. This is the primary backend for production use.

**Architecture:**
```
eBPF Bytecode → LLVM IR → Optimization Passes → Native Code/PTX
```

**Key Components:**
- `llvm_bpf_jit.hpp` - Main JIT compiler class
- `compiler.cpp` - eBPF to LLVM IR compiler
- `optimizer.cpp` - LLVM optimization passes
- `native_code_gen.cpp` - Native code generation
- `ptx_code_gen.cpp` - CUDA PTX generation

**Features:**
- Just-In-Time (JIT) compilation
- Ahead-Of-Time (AOT) compilation  
- CUDA/GPU support via PTX
- Aggressive optimizations
- Helper function inlining

### 4. uBPF Backend (`compat/ubpf-vm/`)

Lightweight interpreter and basic JIT backend based on the uBPF project.

**Features:**
- Simple interpreter mode
- Basic JIT compilation
- Lower memory footprint
- Compatibility fallback

**Limitations:**
- Helper function ID remapping (uBPF supports only 64 helpers)
- Less optimization compared to LLVM backend
- No GPU support

## Design Patterns

### Factory Registration Pattern

VM backends self-register at library load time:

```cpp
// In each VM backend implementation
__attribute__((constructor)) 
static inline void register_llvm_vm_factory() {
    register_vm_factory("llvm", []() -> std::unique_ptr<bpftime_vm_impl> {
        return std::make_unique<llvm_bpf_jit>();
    });
}
```

### Abstract VM Interface

The `bpftime_vm_impl` base class defines the complete VM interface:

```cpp
class bpftime_vm_impl {
public:
    // Core operations
    virtual int load_code(const void *code, size_t code_len) = 0;
    virtual int64_t exec(void *mem, size_t mem_len, int64_t &result) = 0;
    virtual precompiled_ebpf_function compile() = 0;

    // Helper management
    virtual void register_external_function(size_t index, 
                                          const std::string &name,
                                          external_function_t fn) = 0;

    // Advanced features
    virtual std::vector<uint8_t> do_aot_compile() = 0;
    virtual std::optional<std::vector<std::string>> do_get_ptx() = 0;
};
```

### LDDW Helper System

Support for eBPF's Load Double Word instruction with configurable callbacks:

```cpp
struct lddw_helpers {
    int (*map_by_fd)(uint32_t fd);      // FD to map address
    int (*map_by_idx)(uint32_t idx);    // Index to map address
    int (*map_val)(uint64_t map_ptr);   // Map to value address
    int (*var_addr)(uint32_t idx);      // Variable address lookup
    int (*code_addr)(uint32_t idx);     // Code address lookup
};
```

## VM Backends

### Backend Comparison

| Feature | LLVM JIT | uBPF |
|---------|----------|------|
| JIT Compilation | ✓ (Advanced) | ✓ (Basic) |
| AOT Compilation | ✓ | ✗ |
| Interpreter | ✗ | ✓ |
| GPU Support | ✓ | ✗ |
| Helper Inlining | ✓ | ✗ |
| Performance | Excellent | Good |
| Memory Usage | Higher | Lower |
| Compile Time | Slower | Faster |

### Selecting a Backend

```c
// Create VM with specific backend
struct ebpf_vm *vm = ebpf_create("llvm");  // High performance
struct ebpf_vm *vm = ebpf_create("ubpf");  // Lightweight
```

## API Reference

### Core VM Operations

#### Creating and Destroying VMs

```c
// Create a VM instance with specified backend
struct ebpf_vm *ebpf_create(const char *vm_name);

// Destroy a VM instance and free resources
void ebpf_destroy(struct ebpf_vm *vm);
```

#### Loading and Executing Code

```c
// Load eBPF bytecode into the VM
int ebpf_load(struct ebpf_vm *vm, 
              const void *code, 
              uint32_t code_len, 
              char **errmsg);

// Execute loaded program
int64_t ebpf_exec(struct ebpf_vm *vm, 
                  void *mem,        // Memory buffer
                  size_t mem_len,   // Buffer length
                  int64_t *result); // Execution result
```

#### Compilation

```c
// JIT compile to native function
ebpf_jit_fn ebpf_compile(struct ebpf_vm *vm, char **errmsg);

// Enable/disable bounds checking
void ebpf_set_pointer_secret(struct ebpf_vm *vm, uint64_t secret);
void ebpf_toggle_bounds_check(struct ebpf_vm *vm, bool enable);
```

### Helper Function Management

```c
// Register external helper function
int ebpf_register(struct ebpf_vm *vm, 
                  unsigned int idx,      // Helper ID
                  const char *name,      // Function name
                  external_function_t fn); // Function pointer

// Unregister helper function
int ebpf_unregister(struct ebpf_vm *vm, unsigned int idx);
```

### Advanced Operations

```c
// Set LDDW instruction helpers
void ebpf_set_lddw_helpers(struct ebpf_vm *vm,
                          int (*map_by_fd)(uint32_t),
                          int (*map_by_idx)(uint32_t),
                          int (*map_val)(uint64_t),
                          int (*var_addr)(uint32_t),
                          int (*code_addr)(uint32_t));

// AOT compilation
int ebpf_aot_compile(struct ebpf_vm *vm,
                    struct ebpf_aot_options *options,
                    uint8_t **elf_buffer,
                    size_t *elf_size);
```

## Advanced Features

### GPU Execution Support

The LLVM backend supports generating PTX code for NVIDIA GPUs:

```cpp
// Get PTX code for GPU execution
auto ptx_code = vm->do_get_ptx();
if (ptx_code.has_value()) {
    // Use PTX code with CUDA runtime
}
```

### Ahead-Of-Time (AOT) Compilation

Pre-compile eBPF programs to ELF objects:

```cpp
// Generate AOT compiled bytecode
std::vector<uint8_t> aot_bytecode = vm->do_aot_compile();

// Save to file for deployment
std::ofstream out("program.o", std::ios::binary);
out.write((char*)aot_bytecode.data(), aot_bytecode.size());
```

### Inline Map Operations

The LLVM backend supports inlining map operations for zero-overhead access:

```cpp
// Configure inline maps
vm->set_inline_map(map_fd, map_ptr);

// Map operations become direct memory accesses after JIT
```

## Performance Optimizations

### LLVM Optimization Pipeline

The LLVM backend applies sophisticated optimizations:

1. **IR Generation**: Convert eBPF to LLVM IR
2. **Standard Passes**: Apply LLVM's optimization pipeline
3. **eBPF-Specific Passes**:
   - Helper function inlining
   - Map operation optimization
   - Bounds check elimination
4. **Code Generation**: Generate optimized native code

### Memory Management

- **Stack Size**: 512 bytes (configurable via `EBPF_STACK_SIZE`)
- **Max Instructions**: 65536 (configurable via `EBPF_MAX_INSTS`)
- **Max Helpers**: 8192 (configurable)
- **Thread Safety**: Compilation protected by spinlocks

### Performance Tips

1. **Use JIT Compilation**: Always prefer JIT over interpreter
2. **Enable Helper Inlining**: Significant performance boost
3. **Use AOT for Production**: Eliminate runtime compilation
4. **Batch Operations**: Reduce VM creation/destruction overhead
5. **Profile First**: Identify bottlenecks before optimization

## Integration Guide

### Basic Usage Example

```c
#include <ebpf-vm.h>

// eBPF program that returns packet length
uint8_t code[] = {
    0xb7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // r0 = 0
    0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // exit
};

int main() {
    struct ebpf_vm *vm = ebpf_create("llvm");
    char *errmsg;
    
    if (ebpf_load(vm, code, sizeof(code), &errmsg) < 0) {
        fprintf(stderr, "Load failed: %s\n", errmsg);
        free(errmsg);
        return 1;
    }
    
    // JIT compile
    ebpf_jit_fn fn = ebpf_compile(vm, &errmsg);
    if (!fn) {
        fprintf(stderr, "Compile failed: %s\n", errmsg);
        free(errmsg);
        return 1;
    }
    
    // Execute
    uint8_t packet[64] = {0};
    int64_t result = fn(packet, sizeof(packet));
    printf("Result: %ld\n", result);
    
    ebpf_destroy(vm);
    return 0;
}
```

### Integrating with bpftime Runtime

```cpp
// In bpftime runtime
auto vm = std::make_unique<ebpf_vm>("llvm");

// Configure LDDW helpers for map access
vm->set_lddw_helpers(
    [](uint32_t fd) { return map_fd_to_addr(fd); },
    [](uint32_t idx) { return map_idx_to_addr(idx); },
    // ... other helpers
);

// Register runtime helpers
vm->register_external_function(1, "bpf_trace_printk", bpf_trace_printk);
vm->register_external_function(2, "bpf_map_lookup_elem", bpf_map_lookup_elem);
```

## Development Guide

### Adding a New VM Backend

1. **Implement the Interface**:
```cpp
class my_vm : public bpftime::vm::compat::bpftime_vm_impl {
    // Implement all pure virtual methods
};
```

2. **Register the Backend**:
```cpp
__attribute__((constructor))
static void register_my_vm() {
    register_vm_factory("myvm", []() {
        return std::make_unique<my_vm>();
    });
}
```

3. **Add Build Configuration**:
```cmake
add_library(bpftime-vm-myvm STATIC
    myvm/my_vm.cpp
)
target_link_libraries(bpftime-vm-myvm
    bpftime::vm::compat
)
```

### Testing

Run VM tests:
```bash
# Run conformance tests
./build/vm/test/test_conformance

# Run specific backend tests
./build/vm/llvm-jit/test/test_llvm_jit

# Run benchmark tests
./build/vm/benchmark/vm_benchmark
```

### Debugging

Enable debug output:
```bash
# Set environment variables
export BPFTIME_LOG_LEVEL=DEBUG
export LLVM_JIT_DUMP_IR=1  # Dump LLVM IR

# Run with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
```
