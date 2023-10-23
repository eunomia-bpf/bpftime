# Tools

Additional tools for bpftime to improve the user experience.

## bpftime cli

Use for inject dynamic lib of runtime into target process, a wrapper and `LD_PRELOAD` for `bpftime` libraries. This is used as a high-level interface for `bpftime`.

## Install the cli tools an libraries

```bash
sudo apt-get install libelf-dev zlib1g-dev # Install dependencies
cd tools/cli-rs && cargo build --release
mkdir ~/.bpftime && cp ./target/release/bpftime ~/.bpftime
export PATH=$PATH:~/.bpftime
```

## bpftime tool

Inspect or operate the shared memory status of the target process.

