# Tools

Additional tools for bpftime to improve the user experience.

## cli-rs

Use for inject dynamic lib of runtime into target process, and a wrapper and LD_PRELOAD for `bpftime` libraries.

## Install the cli tools an libraries

```bash
sudo apt-get install libelf-dev zlib1g-dev # Install dependencies
cd tools/cli-rs && cargo build --release
mkdir ~/.bpftime && cp ./target/release/bpftime ~/.bpftime
export PATH=$PATH:~/.bpftime
```
