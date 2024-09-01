# ubpf-vm

A wrapper around ubpf to provide unified interface that bpftime could use. There are also some adaptors so we can use some features missing from ubpf
- Remap helper id to 0-63 since ubpf only support 64 helpers
- Patch lddw instructions when loading code
