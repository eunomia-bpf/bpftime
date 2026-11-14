#ifndef BPFTIME_QNX_COMPAT_H
#define BPFTIME_QNX_COMPAT_H

/*
 * QNX Neutrino compatibility notes for bpftime phase-1 (process hook).
 *
 * This header is intentionally lightweight. Prefer using:
 *   #if defined(__QNX__) || defined(BPFTIME_TARGET_QNX)
 * in call sites (same pattern as __APPLE__).
 *
 * Phase-1 assumptions:
 *  - No libbpf / kernel eBPF / syscall-server
 *  - epoll & bpf UAPI types come from runtime/include/bpftime_epoll.h
 *  - Injection via Frida-core only (no LD_PRELOAD / __libc_start_main)
 *  - JIT backend: uBPF
 */

#if defined(__QNX__) || defined(BPFTIME_TARGET_QNX)

#ifndef BPFTIME_ON_QNX
#define BPFTIME_ON_QNX 1
#endif

/* CLOCK_MONOTONIC_COARSE is Linux-specific; helpers use CLOCK_MONOTONIC. */
#ifndef CLOCK_MONOTONIC_COARSE
#define CLOCK_MONOTONIC_COARSE CLOCK_MONOTONIC
#endif

#endif /* __QNX__ || BPFTIME_TARGET_QNX */

#endif /* BPFTIME_QNX_COMPAT_H */
