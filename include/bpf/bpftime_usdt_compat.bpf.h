#ifndef _BPFTIME_USDT_COMPAT_BPF_H
#define _BPFTIME_USDT_COMPAT_BPF_H

/*
 * bpftime USDT compatibility header for BPF_NO_GLOBAL_DATA programs.
 *
 * When BPF_NO_GLOBAL_DATA is defined, __kconfig variables cannot be
 * resolved (no .rodata section is generated), leaving LINUX_HAS_BPF_COOKIE
 * unresolved. The compiler generates both code paths with a runtime branch,
 * causing the LLVM JIT to produce suboptimal native code for bpf_usdt_arg()
 * (~2x USDT overhead vs dead-code-eliminated version).
 *
 * bpftime fully supports bpf_get_attach_cookie, so pre-defining
 * LINUX_HAS_BPF_COOKIE=1 lets the compiler eliminate the slow hash-map
 * path at compile time, producing optimal JIT code.
 *
 * IMPORTANT: This header assumes bpftime (or a kernel with attach-cookie
 * support, i.e. Linux >= 5.15). Do NOT use this header for BPF programs
 * that will be loaded by plain libbpf on older kernels without
 * attach-cookie support — the fast path would be taken unconditionally
 * and bpf_get_attach_cookie() may return 0 or fail. For kernel-portable
 * programs, use the original <bpf/usdt.bpf.h> instead.
 *
 * Usage: Replace #include <bpf/usdt.bpf.h> with this header:
 *
 *     #define BPF_NO_GLOBAL_DATA
 *     #include <vmlinux.h>
 *     #include <bpf/bpf_helpers.h>
 *     #include <bpf/bpftime_usdt_compat.bpf.h>
 *
 * See https://github.com/eunomia-bpf/bpftime/issues/710
 */

#ifdef BPF_NO_GLOBAL_DATA
#define LINUX_HAS_BPF_COOKIE 1
#endif

#include <bpf/usdt.bpf.h>

#endif /* _BPFTIME_USDT_COMPAT_BPF_H */
