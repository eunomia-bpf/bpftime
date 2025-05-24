# ptx program

This is a simple demo program which shows how to call llvmbpf to generate PTX, and uses CUDA driver API to execute the compiled PTX. 

Set `LLVMBPF_CUDA_PATH` to the directory where CUDA installs, for example, `/usr/local/cuda-12.6`

This example demonstrates how to use llvmbpf to generate NVIDIA PTX code from eBPF programs and execute it on CUDA-capable GPUs. It showcases the complete workflow:

1. Initializing LLVM components
2. Defining an eBPF program
3. Compiling the eBPF program to PTX using llvmbpf
4. Compiling the PTX to CUDA binary using NVIDIA's PTX compiler
5. Loading and executing the binary using CUDA Driver API
6. Handling host-device communication for helper functions

## Program Overview

### eBPF to PTX Compilation Flow

The program defines a simple eBPF program that interacts with a BPF map, converts it to PTX, and then runs it on the GPU. This demonstrates how BPF programs can be executed on NVIDIA GPUs with proper support for helper functions.

### Host-Device Communication

The program implements a communication mechanism between the host (CPU) and device (GPU) to handle BPF helper functions. When the BPF program calls a helper function like `map_lookup_elem`, the GPU code signals the host, which processes the request and returns the result.

### Execution Model

1. The eBPF program is compiled to PTX
2. The PTX is wrapped with trampoline code to handle helper function calls
3. NVIDIA's PTX compiler converts the PTX to a CUDA binary
4. The binary is loaded and executed on the GPU
5. A host thread handles helper function requests from the GPU

## Key Components

1. **eBPF Program**: Defined as an array of `ebpf_inst` structures
2. **LLVM JIT Context**: Used to compile eBPF to PTX
3. **PTX Compiler Interface**: Uses NVIDIA's PTX compiler to generate executable code
4. **Shared Memory Structure**: Enables communication between host and device
5. **Helper Function Handlers**: Process requests from the GPU on the host

## Usage

Build and run the example:

```sh
# set the CUDA path, for example, /usr/local/cuda-12.6
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLVMBPF_ENABLE_PTX=1 -DLLVMBPF_CUDA_PATH=/usr/local/cuda-12.6
cmake --build build --target all -j
```

Run the PTX example:

```sh
build/example/ptx/ptx_test
```

## Code Explanation

Here's a detailed explanation of how the `ptx_test.cpp` code works:

### 1. Initialization and Includes

The code starts by including necessary headers for LLVM, CUDA, and eBPF functionality. The missing header files that need to be fixed:

```cpp:llvmbpf/example/ptx/ptx_test.cpp
// LLVM headers are included for JIT compilation
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>

// Project-specific headers need to be properly included
#include <llvm_jit_context.hpp>  // This needs to be fixed
#include <ebpf_inst.h>          // This needs to be fixed

// CUDA headers that need to be correctly included
#include <cuda.h>               // CUDA Driver API
#include <nvPTXCompiler.h>      // NVIDIA PTX Compiler API
#include <cuda_runtime.h>       // CUDA Runtime API
```

### 2. eBPF Program Definition

The code defines a test eBPF program as an array of `ebpf_inst` structures. This program:
1. Takes a memory pointer in `r1`
2. Saves it to `r6`
3. Prepares arguments for a map lookup
4. Calls the `map_lookup_elem` helper function
5. Stores the result to the provided memory
6. Returns 0

```cpp
static const struct ebpf_inst test_prog[] = {
    // r6 = r1 (save input pointer)
    { EBPF_OP_MOV64_REG, 6, 1, 0, 0 },
    
    // Prepare key for map lookup (4 integers: 111,0,0,0)
    { EBPF_OP_MOV64_IMM, 1, 0, 0, 111 },
    { EBPF_OP_STXW, 10, 1, -16, 0 },
    // ... more instructions to prepare arguments ...
    
    // Call map_lookup_elem (helper function 1)
    { EBPF_OP_CALL, 0, 0, 0, 1 },
    
    // Store result to output memory
    { EBPF_OP_LDXW, 1, 0, 0, 0 },
    { EBPF_OP_STXW, 6, 1, 0, 0 },
    
    // Return 0
    { EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
    { EBPF_OP_EXIT, 0, 0, 0, 0 }
};
```

### 3. PTX Compilation Function

The `compile()` function uses NVIDIA's PTX compiler API to convert PTX code to CUDA binary:

```cpp
static std::vector<char> compile(const std::string &ptx)
{
    nvPTXCompilerHandle compiler = NULL;
    // Initialize PTX compiler
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler, 
                                              (size_t)ptx.size(),
                                              ptx.c_str()));
    
    // Set compilation options (e.g., target GPU architecture)
    const char *compile_options[] = { "--gpu-name=sm_60", "--verbose" };
    
    // Compile PTX to binary
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCompile(compiler, 2, compile_options));
    
    // Retrieve compiled binary
    size_t elfSize;
    NVPTXCOMPILER_SAFE_CALL(
        nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
    std::vector<char> elf_binary(elfSize, 0);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
        compiler, (void *)elf_binary.data()));
    
    // Clean up
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
    return elf_binary;
}
```

### 4. Host-Device Communication

The code implements a communication mechanism between host and device using a shared memory structure:

```cpp
struct CommSharedMem {
    int flag1;         // Device -> Host signal
    int flag2;         // Host -> Device signal
    int occupy_flag;   // Lock mechanism
    int request_id;    // Identifies the request type
    long map_id;       // Map identifier
    HelperCallRequest req;    // Request data
    HelperCallResponse resp;  // Response data
    uint64_t time_sum[8];     // Timing information
};
```

### 5. CUDA Kernel Execution

The `elfLoadAndKernelLaunch()` function:
1. Initializes CUDA
2. Loads the compiled binary
3. Sets up shared memory for host-device communication
4. Launches the kernel
5. Starts a host thread to handle helper function calls
6. Waits for kernel completion

```cpp
static int elfLoadAndKernelLaunch(void *elf, size_t elfSize)
{
    // Initialize CUDA
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    
    // Load compiled binary
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    
    // Set up communication channel
    auto comm = std::make_unique<CommSharedMem>();
    memset(comm.get(), 0, sizeof(CommSharedMem));
    
    // Register memory with CUDA for fast access
    CUDA_SAFE_CALL(cuMemHostRegister(comm.get(),
                                   sizeof(CommSharedMem),
                                   CU_MEMHOSTREGISTER_DEVICEMAP));
    
    // Set up map info and shared memory for the kernel
    // ...
    
    // Get kernel function and launch it
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "bpf_main"));
    CUDA_SAFE_CALL(cuLaunchKernel(kernel, 1, 1, 1,  // grid dim
                                1, 1, 1,          // block dim
                                0, nullptr,       // shared mem and stream
                                args, 0));        // arguments
    
    // Start host thread to handle helper calls
    std::thread hostThread([&]() {
        while (!should_exit.load()) {
            if (comm->flag1 == 1) {
                // Process helper function request
                // ...
                comm->flag2 = 1;  // Signal completion
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    // Wait for kernel completion
    CUDA_SAFE_CALL(cuCtxSynchronize());
    hostThread.join();
    
    // Clean up
    // ...
    return 0;
}
```

### 6. Main Function

The `main()` function ties everything together:
1. Sets up signal handler
2. Initializes LLVM components
3. Creates a llvmbpf VM
4. Registers helper functions
5. Loads the eBPF program
6. Generates PTX code
7. Compiles PTX to CUDA binary
8. Executes the binary on the GPU

```cpp
int main()
{
    signal(SIGINT, signal_handler);
    
    // Initialize LLVM components
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    
    // Verify NVPTX target is available
    // ...
    
    // Set up llvmbpf VM
    llvmbpf_vm vm;
    vm.register_external_function(1, "map_lookup", (void *)test_func);
    vm.register_external_function(2, "map_update", (void *)test_func);
    vm.register_external_function(3, "map_delete", (void *)test_func);
    
    // Load eBPF program
    vm.load_code((void *)test_prog, sizeof(test_prog));
    
    // Generate PTX
    llvm_bpf_jit_context ctx(vm);
    auto result = *ctx.generate_ptx(false, "bpf_main", "sm_60");
    
    // Wrap PTX with trampoline code for helper functions
    result = wrap_ptx_with_trampoline(patch_helper_names_and_header(
        patch_main_from_func_to_entry(result)));
    
    // Compile PTX to CUDA binary
    auto bin = compile(result);
    
    // Execute on GPU
    elfLoadAndKernelLaunch(bin.data(), bin.size());
    
    return 0;
}
```

This code demonstrates how to run eBPF programs on NVIDIA GPUs by compiling them to PTX, with support for calling back to the host for helper functions.
