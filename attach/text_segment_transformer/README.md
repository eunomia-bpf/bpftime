# Text Segment Transformer

This component modifies the executable code (text segment) of a process to intercept system calls for bpftime.

## Overview

The Text Segment Transformer rewrites all `syscall` instructions in the program's memory to redirect them to a custom handler. This allows for transparent interception of all system calls without requiring kernel modifications.

## Core Components

- **Text Segment Transformer**: Rewrites executable memory to intercept syscalls
- **Agent Transformer**: Dynamic agent that can be injected into a process to enable syscall interception

## How It Works

1. **Memory Mapping**: The transformer maps a special page at address 0x0 with execute permission
2. **Code Redirection**: It scans all executable memory regions in the process and rewrites `syscall` instructions to jump to the handler
3. **Syscall Dispatch**: When a syscall is executed, it's redirected to the handler function which can:
   - Process the syscall arguments
   - Call registered callbacks (e.g., eBPF programs)
   - Optionally execute the original syscall
   - Return the result to the calling code

## Architecture Details

### Memory Layout

The transformer sets up a special memory layout:
- Maps the first page (0x0) as executable
- Fills the first part with NOPs to preserve the behavior of null pointer dereferences
- Places a jump sequence at a known offset that redirects to the syscall handler

### Syscall Hooking

When a syscall instruction is found in an executable page, it's replaced with:
```asm
call rax  ; Call instruction that jumps to the handler via a register
```

The handler then:
1. Saves the original syscall arguments
2. Converts them from syscall ABI to C function call ABI
3. Calls the dispatch function
4. Converts the result back to syscall ABI format
5. Returns to the original code

## Usage

There are two ways to use the transformer:

### 1. Preload Method

```bash
# Set environment variables
export AGENT_SO=/path/to/bpftime-agent.so
LD_PRELOAD=/path/to/bpftime-agent-transformer.so your_program
```

### 2. Frida Injection

The transformer can also be injected into a running process using Frida.

## Integration with Syscall Trace Attach Implementation

The Text Segment Transformer works together with the Syscall Trace Attach Implementation:

1. The transformer intercepts syscalls at the instruction level
2. It redirects them to the syscall handler
3. The handler calls into the syscall trace implementation
4. The implementation dispatches to registered eBPF programs
5. Results are passed back through the chain 
