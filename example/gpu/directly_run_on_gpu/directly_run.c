/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include "directly_run.skel.h"
#include <inttypes.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

int main(int argc, char **argv)
{
	struct directly_run_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = directly_run_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = directly_run_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	err = directly_run_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

    /* Initialize VecAdd data on GPU maps */
    {
        const uint32_t N = 1024; /* simple default size */
        int map_fd_A = bpf_map__fd(skel->maps.vec_A);
        int map_fd_B = bpf_map__fd(skel->maps.vec_B);
        int map_fd_C = bpf_map__fd(skel->maps.vec_C);
        int map_fd_N = bpf_map__fd(skel->maps.vec_N);

        if (map_fd_A < 0 || map_fd_B < 0 || map_fd_C < 0 || map_fd_N < 0) {
            fprintf(stderr, "Failed to get map fds for vec maps\n");
            goto cleanup;
        }

        /* set N */
        uint32_t key0 = 0;
        uint32_t Nval = N;
        if (bpf_map_update_elem(map_fd_N, &key0, &Nval, BPF_ANY) != 0) {
            fprintf(stderr, "Failed to set vec_N\n");
            goto cleanup;
        }

        /* initialize A and B, zero C */
        for (uint32_t i = 0; i < N; i++) {
            uint32_t key = i;
            uint32_t a = i;
            uint32_t b = 2 * i;
            uint32_t c = 0;
            (void)bpf_map_update_elem(map_fd_A, &key, &a, BPF_ANY);
            (void)bpf_map_update_elem(map_fd_B, &key, &b, BPF_ANY);
            (void)bpf_map_update_elem(map_fd_C, &key, &c, BPF_ANY);
        }
        printf("Initialized VecAdd maps with N=%u\n", N);
    }

    /* Initialize GEMM data on GPU maps: dims={M,N,K}, A[M*K], B[K*N], zero C[M*N] */
    {
        const uint32_t M = 32, Nn = 32, K = 32; /* simple square default */
        int fd_dims = bpf_map__fd(skel->maps.gemm_dims);
        int fd_A = bpf_map__fd(skel->maps.gemm_A);
        int fd_B = bpf_map__fd(skel->maps.gemm_B);
        int fd_C = bpf_map__fd(skel->maps.gemm_C);
        if (fd_dims >= 0 && fd_A >= 0 && fd_B >= 0 && fd_C >= 0) {
            uint32_t key = 0;
            struct { uint32_t m,n,k; } dims = { M, Nn, K };
            (void)bpf_map_update_elem(fd_dims, &key, &dims, BPF_ANY);
            /* Fixed-point Q16.16: A=1.0 -> 1<<16, B=0.01 -> int(0.01*(1<<16)), C=0 */
            const int32_t A_q = (1 << 16);
            const int32_t B_q = (int32_t)(0.01 * (double)(1 << 16));
            const int32_t Z_q = 0;
            for (uint32_t r = 0; r < M; ++r) {
                for (uint32_t c = 0; c < K; ++c) {
                    uint32_t idx = r * K + c;
                    (void)bpf_map_update_elem(fd_A, &idx, &A_q, BPF_ANY);
                }
            }
            for (uint32_t r = 0; r < K; ++r) {
                for (uint32_t c = 0; c < Nn; ++c) {
                    uint32_t idx = r * Nn + c;
                    (void)bpf_map_update_elem(fd_B, &idx, &B_q, BPF_ANY);
                }
            }
            for (uint32_t r = 0; r < M; ++r) {
                for (uint32_t c = 0; c < Nn; ++c) {
                    uint32_t idx = r * Nn + c;
                    (void)bpf_map_update_elem(fd_C, &idx, &Z_q, BPF_ANY);
                }
            }
            printf("Initialized GEMM maps with M=%u N=%u K=%u (Q16.16)\n", M, Nn, K);
        } else {
            fprintf(stderr, "Skipping GEMM init due to missing map fds\n");
        }
    }

    /* Periodically read back a few values for validation */
    {
        int fd_vecC = bpf_map__fd(skel->maps.vec_C);
        int fd_gemmC = bpf_map__fd(skel->maps.gemm_C);
        uint32_t key0 = 0;
        for (;;) {
            if (exiting) break;
            /* VecAdd: print C[0] */
            if (fd_vecC >= 0) {
                uint32_t val32 = 0;
                (void)bpf_map_lookup_elem(fd_vecC, &key0, &val32);
                printf("VecAdd C[0]=%u\n", val32);
            }
            /* GEMM: print C[0] in Q16.16 and as float */
            if (fd_gemmC >= 0) {
                int32_t qval = 0;
                (void)bpf_map_lookup_elem(fd_gemmC, &key0, &qval);
                double fval = (double)qval / 65536.0;
                printf("GEMM C[0]=%d (%.6f)\n", qval, fval);
            }
            fflush(stdout);
            sleep(1);
        }
    }
cleanup:
	/* Clean up */
	directly_run_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
