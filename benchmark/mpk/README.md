# test mpk protect overhead

build the mpk version and put it in another directory, so we have two directories:

- build-mpk
- build

results:

```console
Running benchmarks (100 iterations each)...

Results Summary:
------------------------------------------------------------
Test Name            MPK (ns)        Normal (ns)     Difference     
------------------------------------------------------------
uprobe_uretprobe       228.21        229.44         -1.23 (-0.5%)
  MPK stddev: 27.53
  Normal stddev: 29.70

uretprobe              225.66        222.39         +3.27 (+1.5%)
  MPK stddev: 37.61
  Normal stddev: 36.56

uprobe                 224.26        228.37         -4.11 (-1.8%)
  MPK stddev: 29.71
  Normal stddev: 31.92
```

Seems nearly no difference.
