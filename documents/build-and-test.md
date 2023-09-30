
# Building and test bpftime

## Dependencies

Install the required packages:

```bash
sudo apt install -y --no-install-recommends \
        libelf1 libelf-dev zlib1g-dev make git libboost-dev \
        libboost-program-options-dev binutils-dev
git submodule update --init --recursive
```

### Build and install cli tool

```bash
sudo apt-get install libelf-dev zlib1g-dev # Install dependencies
make build && make install # Build and install the runtime
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

## Testing

Run the test suite to validate the implementation:

```bash
make test
```
