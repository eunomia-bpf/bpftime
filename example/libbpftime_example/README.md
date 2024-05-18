## Example program

This is a sample program showcasing the use of libbpftime.

I am using headers from runtime and I am linking `libbpftime.a` which is installed in `~/.bpftime`


### Prerequisites

following command should have been run, so that `libbpftime.a` exists in `~/.bpftime` directory

```shell
cmake -Bbuild  -DCMAKE_BUILD_TYPE:STRING=Release \
           -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO -DBPFTIME_BUILD_STATIC_LIB=ON
cmake --build build --config Release --target install
```

command to execute the code:
when not using llvm-jit
```shell
g++ -o example main.cpp -I/home/fedora/codes/bpftime-hp/runtime/include/ -I/home/fedora/codes/bpftime-hp/vm/vm-core/compat/include -I/home/fedora/codes/bpftime-hp/vm/compat/include/ -L/home/fedora/.bpftime -lbpftime -lboost_system -lrt -lbpf -L/home/fedora/codes/bpftime-hp/build/vm/ubpf-vm/ubpf/lib -lubpf
```

for `spdlog_exam.cpp`
```shell
g++ -o example spdlog_exam.cpp -I/home/fedora/codes/bpftime-hp/runtime/include -I/home/fedora/codes/bpftime-hp/vm/vm-core/compat/include -I/home/fedora/codes/bpftime-hp/vm/compat/include/ -I/home/fedora/codes/bpftime-hp/third_party/spdlog/include -L/home/fedora/.bpftime -lbpftime -lboost_system -lrt -lbpf
```