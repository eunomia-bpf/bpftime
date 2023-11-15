# Building and run bpftime

## Install Dependencies

Install the required packages:

```bash
sudo apt install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make git libboost1.74-all-dev \
        binutils-dev libyaml-cpp-dev llvm
git submodule update --init --recursive
```

We've tested on Ubuntu 23.04. The recommended `gcc` >= 12.0.0  `clang` >= 16.0.0

On Ubuntu 20.04, you may need to manually switch to gcc-12.

### Build and install cli tool

```bash
sudo apt-get install libelf-dev zlib1g-dev # Install dependencies
make release && make install # Build and install the runtime
cd tools/cli-rs && cargo build --release
mkdir -p ~/.bpftime && cp ./target/release/bpftime ~/.bpftime
export PATH=$PATH:~/.bpftime
```

Then you can run cli:

```console
$ bpftime
Usage: bpftime [OPTIONS] <COMMAND>
...
```

## Compilation for vm

Build the complete runtime:

```bash
make build
```

On old systems, you may need to buil with old binutils version(TODO: fix it):

```bash
make build-old-binutils
```

For a lightweight build without the runtime (only vm library and LLVM JIT):

```bash
make build-vm # build the simple vm with a simple jit
make build-llvm # build the vm with llvm jit
```

## Testing

Run the test suite for runtime to validate the implementation:

```bash
make unit-test
```
