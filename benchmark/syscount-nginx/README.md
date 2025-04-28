# syscount test for nginx

We will test the syscount eBPF program tracing impact on nginx with 5 configurations:

1. No tracing
2. Kernel syscount
3. Kernel syscount do not target nginx pid
4. Userspace syscount
5. Userspace syscount do not target nginx pid



