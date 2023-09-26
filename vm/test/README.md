# Tests

## How to run?

- Build the target `test_Tests` in `PROJECT_ROOT/vm` with `-DBPFTIME_LLVM_JIT=YES`
- Create a venv with Python 3.8, then install packages in `vm/test/requirements.txt`
- Run `pytest -k "test_jit.py and not err-infinite"` in `vm/test/test_frameworks`

## What's expected

Some tests are not expected to succeed. Some of them used features that we haven'e implemented. Some of them are too specilized for other runtimes.

```console
FAILED test_jit.py::test_datafiles[call_unwind.data] - AssertionError: Expected result 0x0, got 0x2, stderr=''

FAILED test_jit.py::test_datafiles[err-call-bad-imm.data] - AssertionError: Expected error 'Failed to load code: invalid call immediate at PC 5', got 'Ext func not found: ext_10000'

FAILED test_jit.py::test_datafiles[err-call-unreg.data] - AssertionError: Expected error 'Failed to load code: call to nonexistent function 63 at PC 5', got 'Ext func not found: ext_0063'

FAILED test_jit.py::test_datafiles[err-endian-size.data] - AssertionError: Expected error 'Failed to load code: invalid endian immediate at PC 0', got 'Unexpected endian size: 48'

FAILED test_jit.py::test_datafiles[err-incomplete-lddw.data] - AssertionError: Expected error 'Failed to load code: incomplete lddw at PC 0', got 'Loaded LDDW at pc=0 which requires an extra pseudo instruction, but the next instruction is not a legal one'

FAILED test_jit.py::test_datafiles[err-incomplete-lddw2.data] - AssertionError: Expected error 'Failed to load code: incomplete lddw at PC 0', got 'Loaded LDDW at pc=0 which requires an extra pseudo instruction, but the next instruction is not a legal one'

FAILED test_jit.py::test_datafiles[err-invalid-reg-dst.data] - AssertionError: Expected error 'Failed to load code: invalid destination register at PC 0', got 'Illegal src reg/dst reg at pc 0'

FAILED test_jit.py::test_datafiles[err-invalid-reg-src.data] - AssertionError: Expected error 'Failed to load code: invalid source register at PC 0', got 'Illegal src reg/dst reg at pc 0'

FAILED test_jit.py::test_datafiles[err-jmp-lddw.data] - AssertionError: Expected error 'Failed to load code: jump to middle of lddw at PC 0', got "Basic Block in function 'bpf_main' does not have terminator!\nlabel %bb_inst_2\nInvalid module generated"

FAILED test_jit.py::test_datafiles[err-jmp-out.data] - AssertionError: Expected error 'Failed to load code: jump out of bounds at PC 0', got 'Instruction at pc=0 is going to jump to an illegal position 3'

FAILED test_jit.py::test_datafiles[err-lddw-invalid-src.data] - AssertionError: Expected error 'Failed to load code: invalid source register for LDDW at PC 0', got "Basic Block in function 'bpf_main' does not have terminator!\nlabel %bb_inst_0\nInvalid module generated"

FAILED test_jit.py::test_datafiles[err-too-many-instructions.data] - AssertionError: Expected error 'Failed to load code: too many instructions (max 65536)', got "Basic Block in function 'bpf_main' does not have terminator!\nlabel %bb_inst_0\nInvalid module generated"

FAILED test_jit.py::test_datafiles[err-unknown-opcode.data] - AssertionError: Expected error 'Failed to load code: unknown opcode 0x06 at PC 0', got 'Unsupported or illegal opcode: 6'
```
