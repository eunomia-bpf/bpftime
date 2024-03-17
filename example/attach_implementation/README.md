# Dynamically loaded attach implementation example

This example demonstrates the usage of bpftime in nginx plugin. It uses eBPF program to filter requests that nginx handles, based on the URL path prefix (i.e, only accepts paths that starts with `/aaa`). The eBPF program will also sends the requests it filtered to a ringbuf, so controller would be able to read them.

## Description

There are three major parts of this example:
- An executable named `controller`. It calls APIs of bpftime_shm to load programs, maps, and links into shared memory, and polls ringbuf output from the eBPF program, and printf them to stdout
- An library named `nginx_plugin_adaptor`. Since nginx uses its own build system, we can't integrate nginx into CMake, so we have this dynamic library to handle things related to bpftime, such as the attach implementation, dynamically registering, call attach context to instantiate handlers, etc
- A nginx module named `ngx_http_bpftime_module`, at folder `nginx_plugin`. This module links `nginx_plugin_adaptor`, and will call several functions implemented in it to achive request filter. This module will be built using nginx's build system

## How to use this example

### Check the nginx version

We assume you are using a nginx version that is compatible with `1.22.1`, if you encounter any incompatible issues when linking or starting nginx, modify the URL at line 34 of CMakeLists.txt at `example/attach_implementation` to your nginx version

### Build controller, adapter with cmake, and use CMake to call the nginx build system

Run the following command at the root directory of the repository:
```console
cmake -DBPFTIME_LLVM_JIT=YES -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_ATTACH_IMPL_EXAMPLE=YES -B build -S .
cmake --build build --config Release --target attach_impl_example_nginx -j$(nproc)
```

### Run and test

We need two terminals, one for the controller, another for running curl

#### Terminal A

Run `build/example/attach_implementation/attach_impl_example_controller /aaaa` at the project root. `/aaaa` is the path prefix that you want to filter. Only paths starts with this string would be accepted, others will be rejected with 403

#### Terminal B

Run `nginx -p $(pwd) -c ./nginx.conf` at `example/attach_implementation` to start nginx. nginx should be started as a daemon process.

Then, run `curl http://127.0.0.1:9023/aaab` and `curl http://127.0.0.1:9023/aaaab` to check the response. You may also find that controller will print accesses that were accepted or rejected.
