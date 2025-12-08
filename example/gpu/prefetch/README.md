# prefetch example

## Usage

Terminal 1:
```
bpftime load ./prefetch
```

Terminal 2:
```
bpftime start ./prefetch_example --kernel=seq_stream --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```

## Benchmark
### P40
Using CPU Hash Map
#### Without prefetch
```
root@mnfe-pve:~/bpftime/example/gpu/prefetch#  ./prefetch_example --kernel=seq_stream --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 22905 MB
Size Factor: 1.5 (oversubscription)
Total Working Set: 34358 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 34358 MB
  Bytes Accessed: 34358 MB
  Median time: 102353 ms
  Min time: 102117 ms
  Max time: 102420 ms
  Bandwidth: 0.351996 GB/s
  Results written to: results.csv

```

#### With Prefetch
```
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 22905 MB
Size Factor: 1.5 (oversubscription)
Total Working Set: 34358 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5

Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig

Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 34358 MB
  Bytes Accessed: 34358 MB
  Median time: 70102 ms
  Min time: 70042.3 ms
  Max time: 70117.9 ms
  Bandwidth: 0.513936 GB/s
  Results written to: results.csv
```

### 5090
Using GPU Array Map
#### Without Prefetch
```
yunwei37@lab:~/mnfe-bpftime-3/example/gpu/prefetch$ ./prefetch_example --kernel=seq_stream --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.5 (oversubscription)
Total Working Set: 48164 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5


Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 48164 MB
  Bytes Accessed: 48164 MB
  Median time: 2166.09 ms
  Min time: 1211.94 ms
  Max time: 2222.68 ms
  Bandwidth: 23.3157 GB/s
  Results written to: results.csv
```
#### With Prefetch
```
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 22905 MB
Size Factor: 1.5 (oversubscription)
Total Working Set: 34358 MB
Stride Bytes: 4096 (page-level)
Kernel: seq_stream
Mode: uvm
Iterations: 5

Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig
Got RunSeqconfig

Results:
  Kernel: seq_stream
  Mode: uvm
  Working Set: 34358 MB
  Bytes Accessed: 34358 MB
  Median time: 70102 ms
  Min time: 70042.3 ms
  Max time: 70117.9 ms
  Bandwidth: 0.513936 GB/s
  Results written to: results.csv
```
