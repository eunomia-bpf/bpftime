#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cuda_neuro_surgeon.bpf.h"


char LICENSE[] SEC("license") = "GPL";
