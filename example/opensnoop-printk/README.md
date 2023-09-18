# opensnoop

Here is a example that demonstrates the usage of userspace syscall trace

## Usage
- Configure & build the project `cmake -S . -B build -G Ninja && cmake --build build --target all --config Debug`
- Run `make -C example/opensnoop-printk`
- Run `LD_PRELOAD=./build/runtime/syscall-server/libbpftime-syscall-server.so ./example/opensnoop-printk/opensnoop`
- Run `LD_PRELOAD=./build/runtime/agent-transformer/libbpftime-agent-transformer.so AGENT_SO=./build/runtime/agent/libbpftime-agent.so ./example/opensnoop-printk/victim` in **another** terminal.
- See output. `open pid=XXXXXX, fname=test.txt, comm=victim` marks a successful running.
