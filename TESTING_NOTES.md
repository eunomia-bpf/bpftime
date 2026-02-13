# Documentation Testing Notes

This branch contains fixes found during documentation testing per the issue:
"按照doc和readme之类的测试一下能不能正常运行，文档有没有不准确的地方，有没有其他 bug？"

## Critical Fix Required

The llvm-jit submodule needs to be updated separately to support LLVM 16-20.

### Changes needed in vm/llvm-jit submodule:
File: `src/llvm_jit_context.cpp` line ~453

Update the candidates list:
```cpp
const char *candidates[] = {
    envSoname && envSoname[0] ? envSoname : (const char *)nullptr,
    "libLLVM.so",
    "libLLVM-20.so",
    "libLLVM-19.so",
    "libLLVM-18.so",
    "libLLVM-17.so",
    "libLLVM-16.so",
    "libLLVM-17.0.6.so",
};
```

### Build requirement:
Enable LLVM preload flag:
```bash
-DBPFTIME_ENABLE_LLVM_PRELOAD=ON
```

## Documentation Fixes Applied
- README.md: Fixed example output (removed "Hello malloc!")
- usage.md: Fixed example output (removed "Hello malloc!")

## Test Results
All installation and usage instructions verified working on Ubuntu 24.04 with LLVM 18.

See /tmp/bpftime-testing-summary.md for full report.
