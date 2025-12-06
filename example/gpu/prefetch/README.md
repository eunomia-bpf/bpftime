# prefetch example

## Usage

Terminal 1:
```
bpftime load ./prefetch
```

Terminal 2:
```
bpftime start ./prefetch_example --kernel=seq_stream --
mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```
