# Usage

Terminal 1
```
BPFTIME_BYPASS_KPROPBE=true  BPFTIME_NOT_LOAD_PATTERN=cuda.*  BPFTIME_RUN_WITH_KERNEL=true bpftime load ./threadhist
```

Terminal 2

```
bpftime start ./vec_add
```

Example output from syscall server

```
21:34:17 
value(0)=10
value(1)=1
21:34:18 
value(0)=20
value(1)=2
21:34:19 
value(0)=30
value(1)=3
```
