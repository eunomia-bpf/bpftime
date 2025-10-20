/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

// VecAdd buffers: A, B, C (u32 elements) and N (length)
struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __u32);
} vec_A SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __u32);
} vec_B SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __u32);
} vec_C SEC(".maps");

	struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u32);
} vec_N SEC(".maps");

static const __u64 (*bpf_get_block_idx)(__u64 *x, __u64 *y, __u64 *z) = (void *)503;
static const __u64 (*bpf_get_block_dim)(__u64 *x, __u64 *y, __u64 *z) = (void *)504;
static const __u64 (*bpf_get_thread_idx)(__u64 *x, __u64 *y, __u64 *z) = (void *)505;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

// GEMM buffers and dims: A (M x K), B (K x N), C (M x N), dims[3]={M,N,K}
struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32[3]);
} gemm_dims SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __s32);
} gemm_A SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __s32);
} gemm_B SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __s32);
} gemm_C SEC(".maps");

SEC("kprobe/__directly_run")
int cuda__run(struct pt_regs *ctx)
{
	bpf_printk("directly run on GPU\n");
	return 0;
}

// Direct-run VecAdd kernel (integer version): C[i] = A[i] + B[i]
SEC("kprobe/__directly_run")
int cuda__vec_add(struct pt_regs *ctx)
{
	__u64 bx = 0, by = 0, bz = 0;
	__u64 bdx = 0, bdy = 0, bdz = 0;
	__u64 tx = 0, ty = 0, tz = 0;

	bpf_get_block_idx(&bx, &by, &bz);
	bpf_get_block_dim(&bdx, &bdy, &bdz);
	bpf_get_thread_idx(&tx, &ty, &tz);

	// 1D index
	__u64 idx64 = bx * bdx + tx;
	__u32 idx = (__u32)idx64;

	__u32 key0 = 0;
	__u32 *Nptr = bpf_map_lookup_elem(&vec_N, &key0);
	if (!Nptr)
		return 0;
	__u32 N = *Nptr;
	if (idx >= N)
		return 0;

	__u32 key = idx;
	__u32 *Ap = bpf_map_lookup_elem(&vec_A, &key);
	__u32 *Bp = bpf_map_lookup_elem(&vec_B, &key);
	__u32 sum = 0;
	if (Ap)
		sum += *Ap;
	if (Bp)
		sum += *Bp;

	bpf_map_update_elem(&vec_C, &key, &sum, (u64)BPF_ANY);
	return 0;
}

// Direct-run GEMM kernel (FP32 baseline): C[M,N] = A[M,K] * B[K,N]
SEC("kprobe/__directly_run")
int cuda__gemm(struct pt_regs *ctx)
{
    __u64 bx = 0, by = 0, bz = 0;
    __u64 bdx = 0, bdy = 0, bdz = 0;
    __u64 tx = 0, ty = 0, tz = 0;
    bpf_get_block_idx(&bx, &by, &bz);
    bpf_get_block_dim(&bdx, &bdy, &bdz);
    bpf_get_thread_idx(&tx, &ty, &tz);

    __u32 key0 = 0;
    __u32 *dims = bpf_map_lookup_elem(&gemm_dims, &key0);
    if (!dims)
        return 0;
    __u32 M = dims[0];
    __u32 N = dims[1];
    __u32 K = dims[2];

    __u32 row = (__u32)(by * bdy + ty);
    __u32 col = (__u32)(bx * bdx + tx);
    if (row >= M || col >= N)
        return 0;

    __s64 acc = 0;
    for (__u32 k = 0; k < K; k++) {
        __u32 a_idx = row * K + k;
        __u32 b_idx = k * N + col;
        __s32 *ap = bpf_map_lookup_elem(&gemm_A, &a_idx);
        __s32 *bp = bpf_map_lookup_elem(&gemm_B, &b_idx);
        __s32 a = ap ? *ap : 0;
        __s32 b = bp ? *bp : 0;
        acc += (__s64)a * (__s64)b;
    }
    __u32 c_idx = row * N + col;
    __s32 out = (__s32)acc;
    (void)bpf_map_update_elem(&gemm_C, &c_idx, &out, (u64)BPF_ANY);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
