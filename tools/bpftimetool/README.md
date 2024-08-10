# bpftime tool

Inspect or operate the shared memory status of the target process.

You can use bpftime tool to serialize the shared memory status to json, and load and run the json file.

## Export data in json

```console
$ ~/.bpftime/bpftimetool export res.json
[2023-10-23 18:45:25.893] [info] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
[2023-10-23 18:45:25.894] [info] bpf_map_handler name=.rodata.str1.1 found at 3
[2023-10-23 18:45:25.894] [info] find prog fd=4 name=do_uprobe_trace
[2023-10-23 18:45:25.894] [info] bpf_perf_event_handler found at 5
INFO [93828]: Global shm destructed
```

## Import data from json

```console
SPDLOG_LEVEL=Debug ~/.bpftime/bpftimetool import /home/yunwei/bpftime/tools/bpftimetool/minimal.json
[2023-10-23 19:02:04.955] [info] Global shm constructed. shm_open_type 3 for bpftime_maps_shm
[2023-10-23 19:02:04.955] [info] import handler fd 3 {"attr":{"btf_id":2,"btf_key_type_id":0,"btf_value_type_id":0,"btf_vmlinux_value_type_id":0,"flags":128,"ifindex":0,"key_size":4,"map_extra":0,"map_type":2,"max_entries":1,"value_size":21},"name":".rodata.str1.1","type":"bpf_map_handler"}
[2023-10-23 19:02:04.955] [info] import handler type bpf_map_handler fd 3
[2023-10-23 19:02:04.956] [info] import handler fd 4 {"attr":{"attach_fds":[5],"cnt":16,"insns":"b701000065642e0a631af8ff0000000018010000756e63200000000063616c6c7b1af0ff0000000018010000746172670000000065745f667b1ae8ff00000000b701000000000000731afcff00000000bfa100000000000007010000e8ffffffb7020000150000008500000006000000b7000000000000009500000000000000","type":0},"name":"do_uprobe_trace","type":"bpf_prog_handler"}
[2023-10-23 19:02:04.956] [info] import handler type bpf_prog_handler fd 4
[2023-10-23 19:02:04.956] [info] import handler fd 5 {"attr":{"_module_name":"example/minimal/victim","offset":4457,"pid":-1,"ref_ctr_off":0,"tracepoint_id":-1,"type":6},"type":"bpf_perf_event_handler"}
[2023-10-23 19:02:04.956] [info] import handler type bpf_perf_event_handler fd 5
INFO [99712]: Global shm destructed
```
