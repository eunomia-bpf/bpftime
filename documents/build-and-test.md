
# Building and test bpftime

## Dependencies

Install the required packages:

```bash
sudo apt install libelf1 libelf-dev zlib1g-dev make git libboost-dev cmake  libboost-program-options-dev
git submodule update --init --recursive
```

## Compilation

Build the complete runtime:

```bash
make build
```

For a lightweight build without the runtime (only core library and LLVM JIT):

```bash
make build-core
make build-llvm
```

## Testing

Run the test suite to validate the implementation:

```bash
make test
```
