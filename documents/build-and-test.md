
# Building and test bpftime

## Dependencies

Install the required packages:

```bash
sudo apt install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make git libboost-dev \
        libboost-program-options-dev binutils-dev
git submodule update --init --recursive
```

## Compilation

Build the complete runtime:

```bash
make build
```

For a lightweight build without the runtime (only vm library and LLVM JIT):

```bash
make build-vm # build the simple vm with a simple jit
make build-llvm # build the vm with llvm jit
```

### Build and install cli tool

```bash
sudo apt-get install libelf-dev zlib1g-dev # Install dependencies
cd tools/cli-rs && cargo build --release
mkdir ~/.bpftime && cp ./target/release/bpftime ~/.bpftime
export PATH=$PATH:~/.bpftime
```

### Build and install runtime

```bash
make install # Install the runtime
```

## Testing

Run the test suite to validate the implementation:

```bash
make test
```
