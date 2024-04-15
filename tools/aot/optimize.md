# Optimize level and runtime configurations in bpftime

There are four major runtime configurations in bpftime:

- Interpreter mode
- LLVM JIT
- ubpf JIT(simple jit)
- LLVM AOT

## LLVM JIT

The baseline configuration for LLVM based JIT.

- Generated LLVM IR from eBPF bytecode. We don't add type information to the IR at this level, so some constraints amybe missing, such as the type of the function, loop bounds, pointer layout, etc.
- Optimized with `-O3` level in the JIT compiled.
- load with the same linker as the AOT runtime.

See https://github.com/eunomia-bpf/bpftime/tree/master/vm/llvm-jit

## ubpf JIT

The baseline configuration for ubpf JIT.

This is a port of ubpf JIT in bpftime.

- Generated Native insts from eBPF bytecode.
- No additional optimization is applied.
- The compile process is faster.

See https://github.com/eunomia-bpf/bpftime/tree/master/vm/ubpf-vm

## LLVM AOT

bpftime AOT engine can compile the LLVM IR from the eBPF bytecode to a `.o` file, or directly load and check a precompiled native object file as the AOT results. It can help us explore better optimization approaches.

The linker(Which help load AOT programs into the runtime) checks:

- The type of the functions and helpers are correct.
- The stack and maps layout are correct.
- All necessary helpers are available in the runtime.

A typical workflow for the AOT process would be:

- The eBPF application create all maps, eBPF prorgams and define which helper set to use. This information is stored in shared memory.
- The relocation of map id or BTF will be performed by the eBPF application with the help of the runtime.
- The verifier checks the eBPF programs, maps, and helpers.
- The AOT compiler `#1` compiles the verified eBPF programs to LLVM IR and Native code, or `#2` based on the type information from the source code, we do some additional process in AST level from the source code with our tools, and compile them to LLVM IR and Native code.
- The AOTed object file is loaded by the runtime with the build-in linker, and can be executed by the runtime as other eBPF programs.

The AOT process will do:

- Compile with `-O3 -fno-stack-protector` and make sure the stack layout is correct.
- Replace the entrance of the eBPF program with function name `bpf_main`.
- All the helpers are replace with corresponding type info and name like `_bpf_helper_ext_0001`

There are three optimizaions in AOT:

- Add Type information and allow the LLVM to do better optimization.
    - Observe: The eBPF bytecode is typeless, so if we JIT from the bytecode only, the LLVM will not be able to do some optimization, such as inlining, loop unrolling, SIMD, etc. With the type information, the LLVM can do better optimization.
    - Approach: Add type information to the LLVM IR or from the source code.
- inline helpers.
    - Observe: Some helpers, such as copy data, strcmp, calc csum, generated random value, are very simple and Don't interact with other maps. They exist because the limitation of verifier. Inline them will avoid the cost of function call and enabled better optimizaion from LLVM.
    - Approach: Prepare a helper implementations in C or LLVM IR, which can be compile and linked with the AOT eBPF res.
    - We could also use better and specific implementation for the helpers.
- inline maps.
    - Observe: some maps are for configurations only, once they are loaded, they will not be read by userspace programs. They are also not shared between different eBPF programs. For example, the `target_pid` filtter in the tracing programs, or some config maps in network programs. Inline them will avoid the cost of map lookup and enabled better optimizaion from LLVM, since the eBPF inst will need helpers or something like `__lddw_helper_map_by_fd` to access them.
    - Approach: inline the maps as global variables in the native code, so that it can be process by the AOT linker. The linker will allocate this `global variables` as the real global variables in the runtime instead of maps, and replace the map access with the address of the global variable directly.


