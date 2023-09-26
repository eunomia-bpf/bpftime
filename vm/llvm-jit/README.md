# LLVM jit for eBPF

A faster and better support on different architectures jit based on LLVM.

## build

```
sudo apt install llvm-dev
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all -j
```

## run

```
build/vm-llvm
```

## Test with bpf-conformance

- Follow the `build` section to build `llvm-jit`
- Follow the instructions to build bpf_conformance_runner

```bash
sudo apt install libboost-dev
git clone https://github.com/Alan-Jowett/bpf_conformance
cd bpf_conformance
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bpf_conformance_runner
```

- Run the tests

```bash
cd bpf_conformance
./build/bin/bpf_conformance_runner  --test_file_directory tests --plugin_path PATH_TO_LLVM_JIT/build/vm-llvm-bpf-test 
```

With `PATH_TO_LLVM_JIT` replaced to the directory of this project

- See the results

If nothing unexpected happens, you will see that `vm-llvm-bpf-test` passes at least 144 tests of the 166 tests. The unpassed tests used features that we haven't supported.

```console
.....
PASS: "tests/stxb-all2.data"
PASS: "tests/stxb-chain.data"
PASS: "tests/stxb.data"
PASS: "tests/stxdw.data"
PASS: "tests/stxh.data"
PASS: "tests/stxw.data"
PASS: "tests/subnet.data"
Passed 147 out of 166 tests.
```
