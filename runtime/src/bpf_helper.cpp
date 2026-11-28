/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"
#include "base_attach_impl.hpp"
#if BPFTIME_HAVE_TRAP_UPROBE_ATTACH
#include "trap_uprobe_attach_impl.hpp"
#endif
#include "handler/map_handler.hpp"
#include "linux/bpf.h"
#include <algorithm>
#include <stdexcept>
#include <system_error>
#if __APPLE__
#include <cstdint>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#elif __linux__
#include <pthread.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#endif
#ifdef BPFTIME_BUILD_WITH_LIBBPF
#include "bpf/bpf.h"
#include "bpf/btf.h"
#include "bpf/libbpf.h"
#include "bpf/libbpf_common.h"
#endif
#include "bpftime_helper_group.hpp"
#include <atomic>
#include <cerrno>
#include <cstdint>
#ifdef ENABLE_BPFTIME_VERIFIER
#include "bpftime-verifier.hpp"
#endif

#include "platform_utils.hpp"
#include "spdlog/spdlog.h"
#include <map>
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <time.h>
#include <unistd.h>
#include <ctime>
#include <filesystem>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_internal.h"
#include "extension/userspace_xdp.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <utility>
#include <bpftime_shm_internal.hpp>
#include <chrono>
#include <thread>

#define PATH_MAX 4096

using namespace std;
bpftime::bpf_attach_ctx &get_global_attach_ctx();
__attribute__((weak)) bpftime::bpf_attach_ctx &get_global_attach_ctx()
{
	SPDLOG_WARN(
		"Calling mocked get_global_attach_ctx, this is not expected");
	throw std::runtime_error(
		"Calling mocked get_global_attach_ctx, this is not expected");
}

#if __linux__ && defined(ENABLE_PROBE_READ_CHECK)
// Event-scoped read-ahead for fault-safe probe reads.
//
// A user-stack walk issues hundreds of small (8/16 byte) probe reads per
// event; with ENABLE_PROBE_READ_CHECK each one costs a process_vm_readv
// syscall. Frame-pointer chains and CFA slots land on nearby, frequently
// overlapping addresses. This window keeps the bytes fetched by the most
// recent widened read so overlapping and adjacent reads are served by memcpy
// without another syscall.
//
// The window is consulted only while an attach callback runs
// (current_thread_attach_callback_depth is raised by attach_callback_scope
// around frida/syscall callbacks; nested tail-call executions stay inside the
// same event). Outside an event every read keeps its original single-syscall
// path and timing, so standalone helper calls are unchanged. The window holds
// a snapshot of userspace bytes for the lifetime of one event; a same-thread
// probe_write_user invalidates it, and fork resets it. No signal handler, BPF
// instruction, caller-facing API or persistent state is introduced.
//
// Coherence contract (M3): reads inside one event observe a single coherent
// snapshot instead of each read independently fetching current memory. A
// store that does not go through bpf_probe_write_user (another thread, or a
// direct BPF map-pointer write) is therefore only visible from the next event.
// This matches the probe model where the callback runs synchronously on the
// hooked thread and the profiled target code is not running concurrently.
//
// Destination contract (M2): source-address fault isolation is unchanged - a
// bad source still returns -EFAULT via the kernel. The hit/fill memcpy writes
// to the BPF program's own destination buffer, which the BPF verifier requires
// to be a valid, large-enough stack/map buffer; a fully fault-safe destination
// is still guaranteed on the out-of-event exact-read path. The trap/SIGTRAP
// signal path never opens an attach scope, so this window never activates there
// (bpf_probe_read is also not in the async-signal-safe helper set).
namespace
{
constexpr size_t PROBE_READ_AHEAD_CAP = 512;

struct probe_read_ahead_window {
	const char *base = nullptr;
	size_t len = 0;
	unsigned long long generation = 0;
	uintptr_t prev_addr = 0;
	char storage[PROBE_READ_AHEAD_CAP];
};

// The window lives in static TLS (this runtime is built with
// -ftls-model=initial-exec), so a probe read never allocates: a heap buffer
// would be allocated inside a helper that can run while the host holds its
// malloc lock. A late dlopen() of the agent must fit its TLS into glibc's
// small static-TLS surplus, which is why the window is capped at 512 bytes
// (maintainer decision, DECISIONS.md 2026-11-27).
thread_local probe_read_ahead_window tls_probe_read_ahead;

static inline probe_read_ahead_window *get_probe_read_ahead_window() noexcept
{
	return &tls_probe_read_ahead;
}

inline bool probe_read_ahead_active()
{
	return bpftime::attach::current_thread_attach_callback_depth > 0;
}

inline void probe_read_ahead_invalidate()
{
	if (auto *window = get_probe_read_ahead_window()) {
		window->base = nullptr;
		window->len = 0;
		window->generation = 0;
	}
}
} // namespace
#endif
extern "C" {

#if __linux__ && \
	(defined(ENABLE_PROBE_READ_CHECK) || defined(ENABLE_PROBE_WRITE_CHECK))
static pid_t probe_access_pid = getpid();
static const int probe_access_atfork_status = pthread_atfork(
	nullptr, nullptr, [] {
		probe_access_pid = getpid();
#if defined(ENABLE_PROBE_READ_CHECK) && __linux__
		probe_read_ahead_invalidate();
#endif
	});
#endif

uint64_t bpftime_override_return(uint64_t ctx, uint64_t value);
uint64_t bpftime_set_retval(uint64_t retval);

uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, ...)
{
	const char *fmt_str = (const char *)fmt;
	va_list args;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	va_start(args, fmt_size);
	long ret = vprintf(fmt_str, args);
#pragma GCC diagnostic pop
	va_end(args);
	return 0;
}

uint64_t bpftime_seq_write(uint64_t, uint64_t, uint64_t)
{
	return -EOPNOTSUPP;
}

long bpftime_strncmp(const char *s1, uint64_t s1_sz, const char *s2)
{
	return strncmp(s1, s2, s1_sz);
}

#ifdef ENABLE_PROBE_READ_CHECK
static size_t probe_page_remaining(const void *address);
static bool probe_read_memory(void *dst, const void *src, size_t size)
{
	struct errno_guard {
		int saved = errno;
		~errno_guard() { errno = saved; }
	} preserve_errno;
	if (size == 0)
		return true;
#if __APPLE__
	mach_vm_size_t copied = 0;
	return mach_vm_read_overwrite(mach_task_self(), (mach_vm_address_t)src,
				      (mach_vm_size_t)size,
				      (mach_vm_address_t)dst,
				      &copied) == KERN_SUCCESS &&
	       copied == size;
#elif __linux__
	pid_t pid = probe_access_atfork_status == 0 ? probe_access_pid :
							  getpid();
	const auto config = std::atomic_ref(bpftime::probe_read_window_config_snapshot)
				    .load(std::memory_order_relaxed);
	const size_t bytes = std::min<size_t>(
		config ? (config & 8191u) - 1 : 512, PROBE_READ_AHEAD_CAP);
	const size_t pre = config ? config >> 13 : 64;
	if (auto *window = get_probe_read_ahead_window();
	    probe_read_ahead_active() && bytes > 0 && window != nullptr) {
		uintptr_t addr = reinterpret_cast<uintptr_t>(src);
		uintptr_t base = reinterpret_cast<uintptr_t>(window->base);
		const auto generation = bpftime::attach::current_thread_attach_event_generation;
		const bool backward = pre && window->generation == generation &&
				      addr < window->prev_addr;
		if (window->generation != generation)
			window->base = nullptr;
		window->prev_addr = addr;
		window->generation = generation;
		if (window->base != nullptr &&
		    addr >= base && addr - base <= window->len &&
		    size <= window->len - (addr - base)) {
			memcpy(dst, window->storage + (addr - base), size);
			return true;
		}
		// Only reads no larger than the configured window (capped at
		// PROBE_READ_AHEAD_CAP = 512 bytes) are widened, and the widened
		// fetch is clamped to the current 4 KiB page; larger reads take the
		// exact path below.
		const size_t offset = addr & 4095u;
		if (size <= bytes && size <= 4096 - offset) {
			const size_t behind = std::min({ backward ? bytes - pre : pre,
						       offset, bytes - size });
			const uintptr_t start = addr - behind;
			const size_t fetch = std::min(bytes, 4096 - (offset - behind));
			struct iovec wide_local = { window->storage, fetch };
			struct iovec wide_remote = { reinterpret_cast<void *>(start),
						    fetch };
			ssize_t got = syscall(SYS_process_vm_readv, pid,
					      &wide_local, 1, &wide_remote, 1,
					      0);
			if (got >= (ssize_t)(behind + size)) {
				window->base = reinterpret_cast<const char *>(start);
				window->len = (size_t)got;
				memcpy(dst, window->storage + behind, size);
				return true;
			}
			window->base = nullptr;
			return false;
		} else {
			window->base = nullptr;
		}
	}
	struct iovec local = { dst, size };
	struct iovec remote = { const_cast<void *>(src), size };
	return syscall(SYS_process_vm_readv, pid, &local, 1, &remote, 1,
		       0) == (ssize_t)size;
#else
#error "Probe memory checks are unsupported on this platform"
#endif
}
#endif

int64_t bpftime_probe_read(uint64_t dst, int64_t size, uint64_t ptr, uint64_t,
			   uint64_t)
{
	if (unlikely(size < 0)) {
		SPDLOG_ERROR("Invalid size: {}", size);
		return -EFAULT;
	}
#ifdef ENABLE_PROBE_READ_CHECK
	return probe_read_memory((void *)dst, (void *)ptr, (size_t)size) ?
		       0 :
		       -EFAULT;
#else
	memcpy((void *)dst, (void *)ptr, (size_t)size);
	return 0;
#endif
}

#if defined(ENABLE_PROBE_WRITE_CHECK) || defined(ENABLE_PROBE_READ_CHECK)
static bool probe_write_memory(void *dst, const void *src, size_t size)
{
	if (size == 0)
		return true;
	// Read-window invalidation is owned by bpftime_probe_write_user() so it
	// applies to both the fault-safe and raw-memcpy write branches.
#if __APPLE__
	mach_vm_size_t copied = 0;
	return mach_vm_read_overwrite(mach_task_self(), (mach_vm_address_t)src,
				      (mach_vm_size_t)size,
				      (mach_vm_address_t)dst,
				      &copied) == KERN_SUCCESS &&
	       copied == size;
#elif __linux__
	struct iovec local = { const_cast<void *>(src), size };
	struct iovec remote = { dst, size };
	return syscall(SYS_process_vm_writev,
		       probe_access_atfork_status == 0 ? probe_access_pid : getpid(),
		       &local, 1, &remote, 1, 0) == (ssize_t)size;
#else
#error "Probe memory checks are unsupported on this platform"
#endif
}
#endif

int64_t bpftime_probe_write_user(uint64_t dst, uint64_t src, int64_t len,
				 uint64_t, uint64_t)
{
	if (unlikely(len < 0)) {
		SPDLOG_ERROR("Invalid len: {}", len);
		return -EFAULT;
	}
#if defined(ENABLE_PROBE_READ_CHECK) && __linux__
	// A same-thread write can change bytes the read-ahead window holds.
	// Do this before the write-check branch so a raw memcpy write (write
	// checking disabled) still cannot make a later read return stale data.
	probe_read_ahead_invalidate();
#endif
#ifdef ENABLE_PROBE_WRITE_CHECK
	return probe_write_memory((void *)dst, (void *)src, (size_t)len) ?
		       0 :
		       -EFAULT;
#else
	memcpy((void *)dst, (void *)src, (size_t)len);
	return 0;
#endif
}

#if defined(__linux__)
// Isolated per-thread PRNG state in initial-exec TLS: never touches __tls_get_addr,
// takes no lock, allocates no heap memory, and leaves host rand() untouched.
static __thread uint64_t bpftime_prng_state
	__attribute__((tls_model("initial-exec"))) = 0;
#else
static thread_local uint64_t bpftime_prng_state = 0;
#endif
static std::atomic<uint64_t> bpftime_prng_seed_seq{ 1 };

static inline uint64_t bpftime_splitmix64(uint64_t z) noexcept
{
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

// Pure userspace seeding (zero active syscalls, preserving host errno):
// combines vDSO CLOCK_MONOTONIC, hardware cycle/virtual counter, TLS/stack ASLR
// addresses, and a monotonic per-seed sequence counter through SplitMix64.
static uint64_t bpftime_prng_seed_userspace() noexcept
{
	const int saved_errno = errno;
	struct timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	errno = saved_errno;

	uint64_t hw = 0;
#if defined(__x86_64__)
	uint32_t lo = 0, hi = 0;
	__asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
	hw = (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
	__asm__ volatile("mrs %0, cntvct_el0" : "=r"(hw));
#endif
	uint64_t z = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
		     static_cast<uint64_t>(ts.tv_nsec);
	z ^= bpftime_splitmix64(hw ^ reinterpret_cast<uintptr_t>(&bpftime_prng_state));
	z ^= bpftime_splitmix64(
		reinterpret_cast<uintptr_t>(&ts) ^
		bpftime_prng_seed_seq.fetch_add(0x9E3779B97F4A7C15ULL,
						std::memory_order_relaxed));
	z = bpftime_splitmix64(z);
	return z != 0 ? z : 0x9E3779B97F4A7C15ULL;
}

uint64_t bpftime_get_prandom_u32()
{
	uint64_t s = bpftime_prng_state;
	if (__builtin_expect(s == 0, 0))
		s = bpftime_prng_seed_userspace();
	s ^= s >> 12;
	s ^= s << 25;
	s ^= s >> 27;
	bpftime_prng_state = s;
	return static_cast<uint32_t>(s * 0x2545F4914F6CDD1DULL);
}

uint64_t bpftime_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t)
{
	timespec spec;
#ifdef __APPLE__
	clock_gettime(CLOCK_MONOTONIC, &spec); // or CLOCK_MONOTONIC_RAW
#else
	clock_gettime(CLOCK_MONOTONIC_COARSE, &spec);
#endif
	return spec.tv_sec * (uint64_t)1000000000 + spec.tv_nsec;
}

uint64_t bpftime_ktime_get_boot_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				   uint64_t)
{
	timespec spec{};
#ifdef CLOCK_BOOTTIME
	if (clock_gettime(CLOCK_BOOTTIME, &spec) == 0)
		return spec.tv_sec * (uint64_t)1000000000 + spec.tv_nsec;
#endif
	clock_gettime(CLOCK_MONOTONIC, &spec);
	return spec.tv_sec * (uint64_t)1000000000 + spec.tv_nsec;
}

#if __linux__
static int get_host_nspid(const char *procfs, const char *entry, int fallback)
{
	if (procfs == nullptr || procfs[0] == '\0')
		return fallback;
	char path[PATH_MAX];
	int length = snprintf(path, sizeof(path), "%s/%s/status", procfs, entry);
	if (length < 0 || static_cast<size_t>(length) >= sizeof(path))
		return fallback;
	FILE *file = fopen(path, "re");
	if (file == nullptr)
		return fallback;
	char *line = nullptr;
	size_t capacity = 0;
	int result = fallback;
	// The Groups line can be arbitrarily long and precedes NSpid in procfs.
	// getline also avoids accepting a PID truncated at a read boundary.
	while (getline(&line, &capacity, file) >= 0) {
		if (strncmp(line, "NSpid:", 6) != 0)
			continue;
		char *end = nullptr;
		errno = 0;
		long value = strtol(line + 6, &end, 10);
		if (errno == 0 && end != line + 6 && value > 0 &&
		    value <= INT32_MAX && (*end == ' ' || *end == '\t' ||
					  *end == '\n' || *end == '\0'))
			result = static_cast<int>(value);
		break;
	}
	free(line);
	fclose(file);
	return result;
}
static thread_local uint64_t bpftime_cached_host_identity = 0;
#endif

// bpf_get_current_pid_tgid caches the identity so it does not pay a syscall on
// every call. The cache cannot be function-local: a forked child inherits the
// parent's cached values, and would report the parent's pid and tid. These live
// at file scope so the fork handler can refresh them in the child.
static pid_t bpftime_cached_tgid = getpid();
#if __linux__
static thread_local int bpftime_cached_tid = -1;
#elif __APPLE__
static thread_local uint64_t bpftime_cached_tid = UINT64_MAX;
#endif

static void bpftime_refresh_cached_identity_in_child()
{
	bpftime_cached_tgid = getpid();
	bpftime_prng_state = 0;
	// tid is meaningless in the child; the next helper call refills it.
#if __linux__
	bpftime_cached_tid = -1;
#elif __APPLE__
	bpftime_cached_tid = UINT64_MAX;
#endif
}

static const int bpftime_identity_atfork_status = pthread_atfork(
	nullptr, nullptr, bpftime_refresh_cached_identity_in_child);

uint64_t bpftime_get_current_pid_tgid(uint64_t, uint64_t, uint64_t, uint64_t,
				      uint64_t)
{
	// If registering the fork handler failed, the cache is still the
	// parent's after a fork, so read the identity directly. The probe
	// access cache above falls back the same way. This exceptional path
	// also recomputes procfs translation on each call.
	if (bpftime_identity_atfork_status != 0) {
#if __linux__
		const char *procfs =
			bpftime::get_agent_bootstrap_config().host_procfs;
		return (uint64_t(get_host_nspid(procfs, "self", getpid())) << 32) |
		       uint32_t(get_host_nspid(procfs, "thread-self",
					      syscall(SYS_gettid)));
#elif __APPLE__
		uint64_t tid = 0;
		pthread_threadid_np(NULL, &tid);
		return ((uint64_t)getpid() << 32) | tid;
#endif
	}
#if __linux__
	if (bpftime_cached_tid == -1) {
		bpftime_cached_tid = static_cast<int>(syscall(SYS_gettid));
		const char *procfs =
			bpftime::get_agent_bootstrap_config().host_procfs;
		bpftime_cached_host_identity =
			(uint64_t(get_host_nspid(procfs, "self", bpftime_cached_tgid)) << 32) |
			uint32_t(get_host_nspid(procfs, "thread-self", bpftime_cached_tid));
	}
#elif __APPLE__
	if (bpftime_cached_tid == UINT64_MAX) {
		pthread_threadid_np(NULL, &bpftime_cached_tid);
	}
#endif
#if __linux__
	return bpftime_cached_host_identity;
#else
	return ((uint64_t)bpftime_cached_tgid << 32) | bpftime_cached_tid;
#endif
}

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) && \
	defined(BPFTIME_BUILD_WITH_LIBBPF)
// Byte offset of member `name` in struct/union `type_id`, looking through
// anonymous members. On success *member_type is the member's resolved type.
// Read the BTF UAPI fields directly instead of libbpf's static inline
// accessors in bpf/btf.h: those calls make GCC record the ExternalProject
// build-tree header (build/libbpf/bpf/btf.h) in this TU's gcno, while the
// integrated-test jobs only receive the gcno/gcda artifacts; lcov 2.x then
// aborts coverage capture because the header is absent there.
static long btf_member_offset(const struct btf *btf, int type_id,
			      const char *name, int *member_type)
{
	const struct btf_type *type = btf__type_by_id(btf, type_id);
	if (type == nullptr ||
	    (BTF_INFO_KIND(type->info) != BTF_KIND_STRUCT &&
	     BTF_INFO_KIND(type->info) != BTF_KIND_UNION))
		return -1;
	const struct btf_member *member =
		(const struct btf_member *)(type + 1);
	const uint32_t member_count = BTF_INFO_VLEN(type->info);
	for (uint32_t i = 0; i < member_count; i++, member++) {
		const uint32_t bit_offset =
			BTF_INFO_KFLAG(type->info) ?
				BTF_MEMBER_BIT_OFFSET(member->offset) :
				member->offset;
		const long offset = (long)(bit_offset / 8);
		const int resolved = btf__resolve_type(btf, member->type);
		const char *member_name =
			btf__name_by_offset(btf, member->name_off);
		if (member_name != nullptr && member_name[0] != '\0') {
			if (strcmp(member_name, name) != 0)
				continue;
			*member_type = resolved;
			return offset;
		}
		const long inner =
			btf_member_offset(btf, resolved, name, member_type);
		if (inner >= 0)
			return offset + inner;
	}
	return -1;
}
#endif

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
// Kernel BTF may hold several task_struct structs (empty forwards on 5.4
// kernels), so scan every non-empty candidate instead of a name lookup.
static const std::pair<size_t, long> &resolve_task_layout()
{
	static const std::pair<size_t, long> layout = []() {
		std::pair<size_t, long> result{ 64 * 1024, -1 };
#ifdef BPFTIME_BUILD_WITH_LIBBPF
		struct btf *btf = btf__parse_raw("/sys/kernel/btf/vmlinux");
		if (btf != nullptr && libbpf_get_error(btf) == 0) {
#if defined(__x86_64__)
			const char *path[] = { "thread", "fsbase" };
#else
			const char *path[] = { "thread", "uw", "tp_value" };
#endif
			for (__u32 id = 1; id < btf__type_cnt(btf); id++) {
				const struct btf_type *type =
					btf__type_by_id(btf, id);
				if (type == nullptr ||
				    BTF_INFO_KIND(type->info) !=
					    BTF_KIND_STRUCT ||
				    BTF_INFO_VLEN(type->info) == 0)
					continue;
				const char *name =
					btf__name_by_offset(btf, type->name_off);
				if (name == nullptr ||
				    strcmp(name, "task_struct") != 0)
					continue;
				int64_t size = btf__resolve_size(btf, (int)id);
				if (size >= (int64_t)sizeof(void *))
					result.first = (size_t)size;
				long offset = 0;
				int type_id = (int)id;
				for (const char *member : path) {
					long m = btf_member_offset(
						btf, type_id, member,
						&type_id);
					if (m < 0) {
						offset = -1;
						break;
					}
					offset += m;
				}
				result.second = offset;
				if (offset >= 0)
					break;
			}
			btf__free(btf);
		}
#endif
		return result;
	}();
	return layout;
}

long bpftime_get_current_task_tpbase_offset() noexcept
{
	long offset = resolve_task_layout().second;
	if (offset >= 0)
		return offset;
#if defined(__x86_64__)
	return 0x14d0;
#else
	return 0x8c0;
#endif
}
#endif

uint64_t bpftime_get_current_task(uint64_t, uint64_t, uint64_t, uint64_t,
				  uint64_t)
{
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
	// Keep the per-thread task storage on the heap through a pthread key
	// instead of a thread_local buffer. This library can be linked into a
	// dlopen'ed module that carries initial-exec TLS; a multi-KB thread_local
	// would land in its static TLS block and exhaust the limited static TLS
	// surplus ("cannot allocate memory in static TLS block").
	// Size and thread-pointer slot of the synthetic task_struct, from kernel
	// BTF when available so programs that read the TLS base through
	// task_struct (thread.fsbase / thread.uw.tp_value) see the real value.
	static const auto &task_layout = resolve_task_layout();
	static const size_t task_storage_size = task_layout.first;
	static pthread_key_t task_key;
	static const int task_key_status = pthread_key_create(&task_key, free);
	if (task_key_status != 0)
		return 0;
	uint8_t *task = static_cast<uint8_t *>(pthread_getspecific(task_key));
	if (task == nullptr) {
		void *raw = nullptr;
		if (posix_memalign(&raw, 64, task_storage_size) != 0)
			return 0;
		task = static_cast<uint8_t *>(raw);
		memset(task, 0, task_storage_size);
		if (pthread_setspecific(task_key, task) != 0) {
			free(task);
			return 0;
		}
	}
	static const size_t tpbase_offset = [] {
		const auto &config = bpftime::get_agent_bootstrap_config();
		if (config.has_system_analysis_tpbase_offset &&
		    config.system_analysis_tpbase_offset <=
			    task_storage_size - sizeof(void *))
			return (size_t)config.system_analysis_tpbase_offset;
		long resolved = bpftime_get_current_task_tpbase_offset();
		return (size_t)resolved <=
				       task_storage_size - sizeof(void *) ?
			       (size_t)resolved :
			       SIZE_MAX;
	}();
	if (tpbase_offset != SIZE_MAX) {
		void *thread_pointer = __builtin_thread_pointer();
		memcpy(task + tpbase_offset, &thread_pointer,
		       sizeof(thread_pointer));
	}
	return (uint64_t)(uintptr_t)task;
#else
	return 0;
#endif
}

uint64_t bpf_get_current_uid_gid(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t)
{
	// Credentials can change after startup, so query them on every call.
	return (static_cast<uint64_t>(getgid()) << 32) |
	       static_cast<uint32_t>(getuid());
}

uint64_t bpftime_ktime_get_ns(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	auto now = std::chrono::steady_clock::now();
	auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
	return ns.time_since_epoch().count();
}

// Copy a task name into the caller buffer with bpf_get_current_comm
// semantics. strncpy alone leaves the destination without a terminator
// when the name fills the whole buffer, so the last byte is forced to
// NUL. A zero-sized buffer must not be touched at all.
void bpftime_copy_comm(char *buf, size_t size, const char *comm)
{
	if (size == 0)
		return;
	strncpy(buf, comm, size);
	buf[size - 1] = '\0';
}

#if __linux__
// Process-wide cache of the executable basename reported by
// bpf_get_current_comm. It deliberately stays in BSS with pthread_once
// publication so every failure mode of the previous std::string cache
// (removed by #723) is absent:
//  - no heap allocation on any call, so an uprobe on malloc/mmap/brk cannot
//    re-enter a held allocator lock (the first call stays allocation-free,
//    covered by the [reentry] unit test);
//  - pthread_once orders the first publication, so concurrent first calls
//    cannot race on the buffer;
//  - execve replaces the whole address space and rebuilds this BSS state, so
//    the new image always resolves its own name; fork inherits an equally
//    valid name because /proc/self/exe does not change across fork.
// The init routine performs only a raw readlink syscall and string scans, so
// a thread blocked in pthread_once cannot wait on an allocator lock.
//
// pthread_once is not async-signal-safe: the cache relies on being called from
// ordinary thread context (the frida backend). Trap-backend BPF links run in
// SIGTRAP context and admit only an audited async-signal-safe helper set
// (see attach/trap_uprobe_attach_impl/README.md), so this helper must be
// re-audited before ever being admitted there.
//
// A fork concurrent with another thread running the once routine would let
// the child inherit an in-progress control word with no thread to finish it.
// The atfork child handler resets the cache, mirroring the pid/tgid cache
// handling from #812.
namespace
{
struct current_comm_cache {
	pthread_once_t once = PTHREAD_ONCE_INIT;
	char basename[PATH_MAX] = {};
	bool valid = false;
};

current_comm_cache &get_current_comm_cache()
{
	// constinit turns the BSS/no-dynamic-init requirement above into a
	// compile-time guarantee: no guard variable and no atexit registration.
	constinit static current_comm_cache cache;
	return cache;
}

void initialize_current_comm_cache()
{
	auto &cache = get_current_comm_cache();
	char path[PATH_MAX];
	auto len = readlink("/proc/self/exe", path, sizeof(path) - 1);
	if (len == -1)
		return;
	path[len] = '\0';
	const char *filename = path;
	for (decltype(len) index = 0; index < len; ++index) {
		if (path[index] == '/')
			filename = path + index + 1;
	}
	strncpy(cache.basename, filename, sizeof(cache.basename) - 1);
	cache.basename[sizeof(cache.basename) - 1] = '\0';
	cache.valid = true;
}

void reset_current_comm_cache_in_child()
{
	// The child has a single thread, so re-arming the once control word
	// cannot race with a waiter.
	auto &cache = get_current_comm_cache();
	cache.once = PTHREAD_ONCE_INIT;
	cache.valid = false;
}

const int current_comm_atfork_status =
	pthread_atfork(nullptr, nullptr, reset_current_comm_cache_in_child);
} // namespace
#endif

uint64_t bpftime_get_current_comm(uint64_t buf, uint64_t size, uint64_t,
				  uint64_t, uint64_t)
{
#if __linux__
	// If the fork handler could not be registered, a forked child might keep
	// an in-progress once word, so bypass the cache and resolve every call
	// directly, exactly as #723 did.
	if (current_comm_atfork_status == 0) {
		auto &cache = get_current_comm_cache();
		pthread_once(&cache.once, initialize_current_comm_cache);
		if (cache.valid) {
			bpftime_copy_comm((char *)(uintptr_t)buf, (size_t)size,
					  cache.basename);
			return 0;
		}
	}
	// First resolution failed (for example /proc unavailable), or the atfork
	// handler registration failed: keep the historical per-call retry and
	// its return code.
#endif
	// This helper can run from an allocation callback (for example an
	// uprobe on malloc/mmap) while libc already holds an allocator lock.
	// A lazily initialized std::string takes the heap on its first long
	// value, which re-enters the allocator and can self-deadlock. The
	// fallback path resolves the name into a stack buffer without heap.
	char strbuf[PATH_MAX];
	auto len = readlink("/proc/self/exe", strbuf, std::size(strbuf) - 1);
	if (len == -1)
		return 1;
	strbuf[len] = 0;
	const char *filename = strbuf;
	for (decltype(len) index = 0; index < len; ++index) {
		if (strbuf[index] == '/')
			filename = strbuf + index + 1;
	}
	bpftime_copy_comm((char *)(uintptr_t)buf, (size_t)size, filename);
	return 0;
}
// Userspace map backends report failure the libc way: -1 with the host errno
// set. The kernel's bpf_map_*_elem helpers report a negative errno in the
// return value instead, and BPF programs branch on -EEXIST/-ENOENT/-E2BIG.
// Translate here, at the BPF-helper boundary only: the syscall-facing shims
// keep the -1 + errno contract. A backend that already returns a specific
// kernel errno (for example -ENOTSUP from an unsupported operation) passes
// through unchanged.
static uint64_t kernel_map_helper_result(long result)
{
	if (result >= 0)
		return 0;
	if (result != -1)
		return (uint64_t)result;
	return (uint64_t)-(errno != 0 ? errno : EINVAL);
}

uint64_t bpftime_map_lookup_elem_helper(uint64_t map, uint64_t key, uint64_t,
					uint64_t, uint64_t)
{
	return (uint64_t)bpftime::shm_holder.global_shared_memory
		.bpf_map_lookup_elem((int)map, (void *)key, false);
}

uint64_t bpftime_map_update_elem_helper(uint64_t map, uint64_t key,
					uint64_t value, uint64_t flags,
					uint64_t)
{
	return kernel_map_helper_result(
		bpftime::shm_holder.global_shared_memory.bpf_map_update_elem(
			(int)map, (void *)key, (void *)value, flags, false));
}

uint64_t bpftime_map_delete_elem_helper(uint64_t map, uint64_t key, uint64_t,
					uint64_t, uint64_t)
{
	return kernel_map_helper_result(
		bpftime::shm_holder.global_shared_memory.bpf_delete_elem(
			(int)map, (void *)key, false));
}

uint64_t bpftime_map_push_elem_helper(uint64_t map, uint64_t value,
				      uint64_t flags, uint64_t, uint64_t)
{
	return kernel_map_helper_result(
		bpftime::shm_holder.global_shared_memory.bpf_map_push_elem(
			(int)map, (void *)value, flags, false));
}

uint64_t bpftime_map_pop_elem_helper(uint64_t map, uint64_t value, uint64_t,
				     uint64_t, uint64_t)
{
	return kernel_map_helper_result(
		bpftime::shm_holder.global_shared_memory.bpf_map_pop_elem(
			(int)map, (void *)value, false));
}

uint64_t bpftime_map_peek_elem_helper(uint64_t map, uint64_t value, uint64_t,
				      uint64_t, uint64_t)
{
	return kernel_map_helper_result(
		bpftime::shm_holder.global_shared_memory.bpf_map_peek_elem(
			(int)map, (void *)value, false));
}

#ifdef ENABLE_PROBE_READ_CHECK
static size_t probe_page_remaining(const void *address)
{
	static const size_t page_size = []() {
		long result = sysconf(_SC_PAGESIZE);
		return result > 0 ? (size_t)result : (size_t)4096;
	}();
	return page_size - ((uintptr_t)address % page_size);
}

static void clear_probe_memory(void *dst, size_t size)
{
	static const char zeros[256] = {};
	auto output = (char *)dst;
	while (size > 0) {
		size_t chunk_size = std::min(size, sizeof(zeros));
		if (!probe_write_memory(output, zeros, chunk_size))
			return;
		output += chunk_size;
		size -= chunk_size;
	}
}
#endif

int64_t bpf_probe_read_str(uint64_t buf, uint64_t bufsz, uint64_t ptr, uint64_t,
			   uint64_t)
{
	if (bufsz == 0)
		return 0;

	auto dst = (char *)(uintptr_t)buf;
	auto src = (const char *)(uintptr_t)ptr;
#ifndef ENABLE_PROBE_READ_CHECK
	for (size_t i = 0; i < (size_t)bufsz; i++) {
		if (src[i] == '\0') {
			dst[i] = '\0';
			return (int64_t)i + 1;
		}
		if (i == (size_t)bufsz - 1) {
			dst[i] = '\0';
			return (int64_t)bufsz;
		}
		dst[i] = src[i];
	}
	return (int64_t)bufsz;
#else
	char scratch[256];
	size_t copied = 0;
	while (copied < (size_t)bufsz) {
		size_t chunk_size =
			std::min(sizeof(scratch), (size_t)bufsz - copied);
		chunk_size = std::min(chunk_size,
				      probe_page_remaining(src + copied));
		if (!probe_read_memory(scratch, src + copied, chunk_size)) {
			clear_probe_memory(dst, (size_t)bufsz);
			return -EFAULT;
		}

		auto terminator = (char *)memchr(scratch, '\0', chunk_size);
		bool complete = terminator != nullptr;
		if (complete) {
			chunk_size = (size_t)(terminator - scratch) + 1;
		} else if (chunk_size == (size_t)bufsz - copied) {
			scratch[chunk_size - 1] = '\0';
			complete = true;
		}

		if (!probe_write_memory(dst + copied, scratch, chunk_size))
			return -EFAULT;
		copied += chunk_size;
		if (complete)
			return (int64_t)copied;
	}
	return (int64_t)copied;
#endif
}

uint64_t bpf_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t)
{
	struct timespec ts;
#if __APPLE__
	clock_gettime(CLOCK_MONOTONIC, &ts); // or CLOCK_MONOTONIC_RAW
#else
	clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
#endif
	return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

uint64_t bpf_ringbuf_output(uint64_t rb, uint64_t data, uint64_t size,
			    uint64_t flags, uint64_t)
{
	int fd = (int)rb;
	if (flags & ~(BPF_RB_NO_WAKEUP | BPF_RB_FORCE_WAKEUP))
		return (uint64_t)-EINVAL;
	auto buf = bpftime_ringbuf_reserve(fd, size);
	if (!buf) {
		SPDLOG_ERROR("Failed to reserve when executing ringbuf output");
		return (uint64_t)-1;
	}
	memcpy(buf, (const void *)(uintptr_t)data, size);
	bpftime_ringbuf_submit(fd, buf, false);
	return 0;
}

uint64_t bpf_ringbuf_reserve(uint64_t rb, uint64_t size, uint64_t flags,
			     uint64_t, uint64_t)
{
	int fd = (int)rb;
	if (flags != 0) {
		SPDLOG_WARN(
			"Currently only supports ringbuf_reserve with flags=0");
	}
	return (uint64_t)(uintptr_t)bpftime_ringbuf_reserve(fd, size);
}

uint64_t bpf_ringbuf_submit(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			    uint64_t)
{
	int32_t *ptr = (int32_t *)(uintptr_t)data;
	int fd = ptr[-1];
	// As in the kernel, the record is committed whatever the flags say.
	// BPF_RB_NO_WAKEUP and BPF_RB_FORCE_WAKEUP only steer the consumer
	// wakeup, which the polling consumer does without. A log line here
	// would write on every submit of programs that pass them.
	(void)flags;
	bpftime_ringbuf_submit(fd, (void *)(uintptr_t)data, false);
	return 0;
}

uint64_t bpf_ringbuf_discard(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			     uint64_t)
{
	int32_t *ptr = (int32_t *)(uintptr_t)data;
	int fd = ptr[-1];
	// As in the kernel, the record is committed whatever the flags say.
	// BPF_RB_NO_WAKEUP and BPF_RB_FORCE_WAKEUP only steer the consumer
	// wakeup, which the polling consumer does without. A log line here
	// would write on every submit of programs that pass them.
	(void)flags;
	bpftime_ringbuf_submit(fd, (void *)(uintptr_t)data, true);
	return 0;
}

uint64_t bpf_perf_event_output(uint64_t ctx, uint64_t map, uint64_t flags,
			       uint64_t data, uint64_t size)
{
	int32_t current_cpu = bpftime_get_current_cpu();
	if (unlikely(current_cpu == -1)) {
		SPDLOG_DEBUG(
			"Unable to get current cpu when running perf event output");
		return (uint64_t)-1;
	}
	// Software perf events are sharded by (pid, tid) and the consumer polls
	// every slot, so no per-cpu invariant has to be emulated in userspace.
	// Pinning the thread to the current cpu around each output cost three
	// scheduler syscalls (getaffinity + two setaffinity) and forced two
	// migrations per event for no correctness benefit. The cpu index comes
	// from the vDSO via bpftime_get_current_cpu(); a thread migrating between calls
	// only selects a different array slot, which the consumer already
	// enumerates.
	int fd = (int)map;
	// Check map type. userspace perf event array, or shared perf event
	// array?
	bpftime::bpf_map_type map_ty;
	if (int err = bpftime_map_get_info(fd, nullptr, nullptr, &map_ty);
	    err < 0) {
		SPDLOG_ERROR("Unable to query map type of fd {}", fd);
		return (uint64_t)-1;
	}
	if (map_ty == bpftime::bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY) {
		const int32_t *val_ptr =
			(int32_t *)(uintptr_t)bpftime::shm_holder
				.global_shared_memory.bpf_map_lookup_elem(
					fd, &current_cpu, false);
		if (val_ptr == nullptr) {
			SPDLOG_ERROR("Invalid map fd for perf event output: {}",
				     fd);
			errno = EINVAL;
			return (uint64_t)-1;
		}
		int32_t perf_handler_fd =
			__atomic_load_n(val_ptr, __ATOMIC_ACQUIRE);
		return (uint64_t)bpftime_perf_event_output(
			perf_handler_fd, (const void *)(uintptr_t)data,
			(size_t)size);
	}
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
	else if (map_ty == bpftime::bpf_map_type::
				   BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY) {
		return (uint64_t)bpftime_shared_perf_event_output(
			fd, (const void *)(uintptr_t)data, (size_t)size);
	}
#endif
	else {
		SPDLOG_ERROR(
			"Attempting to run perf_output on a non-perf array map");
		return (uint64_t)-1;
	}
}

// Tail-call targets always receive a 64-byte context buffer.
//
// The helper's ctx may be either host-owned memory from the region passed
// to bpftime_prog::bpftime_prog_exec (whose pointer and size are published
// per thread), or a replacement context the running BPF program built on
// its own BPF stack and passed as r1. The BPF ABI carries no size for the
// latter, so only pointers contained in the host entry region are bounded;
// replacement contexts keep the historical full 64-byte copy.
//
// Safety of a replacement context (the pointer actually lands on the
// program's BPF stack and 64 bytes are readable there) is a verifier/ABI
// precondition this helper cannot enforce without BPF-stack metadata; a
// malformed program pointing r1 outside its stack frame is tracked under
// the verifier-bypass work (issue #699).
static void bpftime_prepare_tail_call_context(char (&context)[64],
					      uint64_t ctx)
{
	constexpr size_t tail_call_context_size = 64;
	if (ctx == 0) {
		memset(context, 0, tail_call_context_size);
		return;
	}
	const uintptr_t ctx_addr = (uintptr_t)ctx;
	size_t copy_size = tail_call_context_size;
	if (bpftime::current_thread_exec_ctx.has_value()) {
		const auto &entry = *bpftime::current_thread_exec_ctx;
		const uintptr_t entry_begin =
			reinterpret_cast<uintptr_t>(entry.ptr);
		const uintptr_t entry_end = entry_begin + entry.size;
		if (ctx_addr >= entry_begin && ctx_addr < entry_end) {
			// Inside the host entry region (exact base or an
			// interior alias such as entry + 4): only the bytes
			// from the alias to the region end are known valid.
			copy_size = std::min(
				static_cast<size_t>(entry_end - ctx_addr),
				tail_call_context_size);
		}
		// Fully outside the region: a program-owned replacement
		// context; copy all 64 bytes as the ABI historically does.
	} else {
		// Helper reached without any published exec context: the
		// real size is unknown, so keep the historical 64-byte copy
		// (still capped to the destination buffer) and warn once.
		static std::once_flag tail_call_ctx_info_warn_once;
		std::call_once(tail_call_ctx_info_warn_once, [] {
			SPDLOG_WARN(
				"tail_call: exec context info is unavailable; copying the full 64-byte tail-call context");
		});
	}
	memcpy(context, reinterpret_cast<const void *>(ctx_addr),
	       copy_size);
	if (copy_size < tail_call_context_size) {
		memset(context + copy_size, 0,
		       tail_call_context_size - copy_size);
	}
}

uint64_t bpftime_tail_call(uint64_t ctx, uint64_t prog_array, uint64_t index)
{
	int fd = (int)prog_array;
	if (!bpftime_is_prog_array(fd)) {
		SPDLOG_ERROR("Expected fd {} to be a prog array fd", fd);
		return -1;
	}
	int idx = index;

	int *to_call_id_ptr = (int *)bpftime_map_lookup_elem(fd, &idx);
	if (!to_call_id_ptr) {
		SPDLOG_ERROR("Unable to lookup index {} of prog array {}", idx,
			     fd);
		return -1;
	}
	int to_call_fd = *to_call_id_ptr;
	SPDLOG_DEBUG("tail call helper: calling prog fd {}", to_call_fd);

	if (bpftime_is_prog_fd(to_call_fd)) {
		constexpr uint32_t MAX_TAIL_CALL_CNT = 32;
		static thread_local uint32_t tail_call_depth = 0;
		if (tail_call_depth >= MAX_TAIL_CALL_CNT) {
			SPDLOG_ERROR("tail call depth limit exceeded");
			return -1;
		}
		struct tail_call_depth_guard {
			uint32_t &depth;
			explicit tail_call_depth_guard(uint32_t &depth)
				: depth(depth)
			{
				depth++;
			}
			~tail_call_depth_guard()
			{
				depth--;
			}
		} guard(tail_call_depth);

		bpftime::bpftime_prog *prog = nullptr;
		try {
			prog = get_global_attach_ctx().find_instantiated_prog(
				to_call_fd);
		} catch (const std::exception &) {
		}
		std::optional<bpftime::bpftime_prog> fallback;
		if (prog == nullptr) {
			const auto &handler = std::get<bpftime::bpf_prog_handler>(
				bpftime::shm_holder.global_shared_memory.get_handler(
					to_call_fd));
			fallback.emplace(handler.insns.data(), handler.insns.size(),
					 handler.name.c_str());
			prog = &*fallback;
			if (bpftime::load_prog_and_helpers(
				    prog,
				    bpftime::bpftime_get_runtime_config()) < 0)
				return -1;
		}

		char context[64];
		bpftime_prepare_tail_call_context(context, ctx);
		uint64_t retval = 0;
		int err = prog->bpftime_prog_exec(context, sizeof(context),
						  &retval);
		if (err < 0) {
			SPDLOG_ERROR(
				"Failed to execute userspace tail call target fd {}",
				to_call_fd);
			return -1;
		}
		bpftime::current_thread_tail_call_ret = retval;
		return 0;
	}

#if __linux__ && defined(BPFTIME_BUILD_WITH_LIBBPF)
	char context[64];
	bpftime_prepare_tail_call_context(context, ctx);
	LIBBPF_OPTS(bpf_test_run_opts, run_opts, .ctx_in = context,
		    // .ctx_out = context_out,
		    .ctx_size_in = sizeof(context),
		    // .ctx_size_out = sizeof(context_out)
	);
	int err = bpf_prog_test_run_opts(to_call_fd, &run_opts);
	if (err < 0) {
		close(to_call_fd);
		SPDLOG_ERROR("Failed to run kernel program: {}", errno);
		return -1;
	}
	close(to_call_fd);
	bpftime::current_thread_tail_call_ret = run_opts.retval;
	return 0;
#else
	SPDLOG_ERROR(
		"tail_call to kernel program fd {} requires libbpf support",
		to_call_fd);
	return -ENOTSUP;
#endif
}

uint64_t bpftime_get_attach_cookie(uint64_t ctx, uint64_t, uint64_t, uint64_t,
				   uint64_t)
{
	if (auto *signal_ctx = bpftime::attach::current_signal_callback)
		return signal_ctx->cookie.value_or(0);
	uint64_t cookie;
	if (bpftime_get_current_thread_cookie(&cookie)) {
		SPDLOG_DEBUG("Get cookie: {}", cookie);
		return cookie;
	} else {
		SPDLOG_DEBUG("Cookie doesn't exist");
		return 0;
	}
}

uint64_t bpftime_get_func_ip(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	if (auto *ctx = bpftime::attach::current_signal_callback)
		return ctx->function_ip;
	return bpftime::attach::current_thread_attach_func_ip;
}

uint64_t bpftime_get_smp_processor_id()
{
	int cpu = bpftime_get_current_cpu();
	if (cpu == -1) {
		SPDLOG_DEBUG("sched_getcpu error");
		return 0; // unlikely
	}
	return (uint64_t)cpu;
}

// From https://github.com/microsoft/ebpf-for-windows
int64_t bpftime_csum_diff(const void *from, int from_size, const void *to,
			  int to_size, int seed)
{
	int csum_diff = -EINVAL;

	if ((from_size % 4 != 0) || (to_size % 4 != 0)) {
		// size of buffers should be a multiple of 4.
		goto Exit;
	}

	csum_diff = seed;
	if (to != NULL) {
		for (int i = 0; i < to_size / 2; i++) {
			csum_diff += (uint16_t)(*((uint16_t *)to + i));
		}
	}
	if (from != NULL) {
		for (int i = 0; i < from_size / 2; i++) {
			csum_diff += (uint16_t)(~*((uint16_t *)from + i));
		}
	}

	// Adding 16-bit unsigned integers or their one's complement will
	// produce a positive 32-bit integer, unless the length of the buffers
	// is so long, that the signed 32 bit output overflows and produces a
	// negative result.
	if (csum_diff < 0) {
		csum_diff = -EINVAL;
	}
Exit:
	return csum_diff;
}

#define ETH_HLEN 14 /* Total octets in header.	 */

long bpftime_xdp_adjust_head(struct xdp_md_userspace *xdp, int offset)
{
	// We don't use xdp meta data
	uint64_t data = xdp->data + offset;
	if (unlikely(data > xdp->data_end - ETH_HLEN) || data > xdp->buffer_end)
		return -EINVAL;
	if (data < xdp->buffer_start) {
		// The requested head is before the buffer, so the head can only
		// move as far as buffer_start. Move the packet there so that the
		// bytes xdp->data is about to point at are the packet, and so the
		// move stays inside [buffer_start, buffer_end): the old form
		// computed the destination as
		// buffer_start + (buffer_start - data), which both lands past the
		// requested head and can run beyond buffer_end.
		uint64_t len = xdp->data_end - xdp->data;
		if (len > xdp->buffer_end - xdp->buffer_start)
			return -EINVAL;
		memmove(reinterpret_cast<void *>(xdp->buffer_start),
			reinterpret_cast<void *>(xdp->data), len);
		data = xdp->buffer_start;
	}
	xdp->data = data;
	return 0;
}

long bpftime_xdp_adjust_tail(struct xdp_md_userspace *xdp_md, int delta)
{
	// We don't use xdp meta data
	uint64_t data = xdp_md->data_end + delta;
	if (data < xdp_md->data || data < xdp_md->buffer_start ||
	    data > xdp_md->buffer_end) {
		return -EINVAL;
	}
	xdp_md->data_end = data;
	return 0;
}

long bpftime_xdp_load_bytes(struct xdp_md_userspace *xdp_md, __u32 offset,
			    void *buf, __u32 len)
{
	// We don't support fragmented packets
	uint64_t data = xdp_md->data + offset;
	if (data + len > xdp_md->data_end) {
		return -EINVAL;
	}
	memcpy(buf, reinterpret_cast<void *>(data), len);
	return 0;
}
using namespace bpftime;

int64_t bpftime_get_stackid(uint64_t ctx_raw, uint64_t map_raw, uint64_t flags,
			    uint64_t, uint64_t)
{
	if (!(flags & BPF_F_USER_STACK)) {
		SPDLOG_ERROR(
			"bpftime_get_stackid only supports collect user stack!");
		return -ENOTSUP;
	}

	const int ATTACH_UPROBE = 6;

	auto &attach_ctx = get_global_attach_ctx();
	auto attach_impl =
		attach_ctx.get_attach_impl_by_attach_type(ATTACH_UPROBE);
	if (!attach_impl.has_value()) {
		SPDLOG_ERROR(
			"Unable to get stack id: frida_uprobe_attach_impl not registered");
		return -ENOTSUP;
	}
	attach::stack_walk_data walk;
	(void)(*attach_impl)
		->call_attach_specific_function("generate_stack", &walk);
	// No frames is normal when the probe has no unwinding context, such as a
	// syscall-trace or tracepoint program whose pt_regs was synthesized:
	// treat it as an empty stack without allocating on the host heap (which
	// can deadlock if the syscall fired while libc malloc held its arena lock).
	const uint64_t *frames = walk.frames;
	size_t frame_count = walk.count;

	auto frames_to_skip = static_cast<size_t>(flags & BPF_F_SKIP_FIELD_MASK);
	SPDLOG_DEBUG("Skipping {} frames", frames_to_skip);
	if (frames_to_skip >= frame_count) {
		frame_count = 0;
	} else {
		frames += frames_to_skip;
		frame_count -= frames_to_skip;
	}
	SPDLOG_DEBUG("After skipping, collected {} frames", frame_count);
	int real_map_fd = (int)map_raw;
	auto &map_handler = std::get<bpf_map_handler>(
		shm_holder.global_shared_memory.get_handler(real_map_fd));

	robust_shm_trylock_guard guard(map_handler.get_map_lock());
	if (!guard.owns_lock())
		return -EBUSY;
	auto impl = *shm_holder.global_shared_memory.try_get_stack_trace_impl(
		real_map_fd);
	int ret = impl->fill_stack_trace(frames, frame_count,
					 flags & BPF_F_REUSE_STACKID,
					 flags & BPF_F_FAST_STACK_CMP);

	SPDLOG_DEBUG("Got stackid {}", ret);
	// Keep negative errno values (e.g. -EEXIST on a collision) intact;
	// valid stack ids are non-negative and unchanged by the conversion.
	return ret;
}

int64_t bpftime_get_stack(uint64_t ctx_raw, uint64_t buf, uint64_t size,
			  uint64_t flags, uint64_t)
{
	if (!(flags & BPF_F_USER_STACK)) {
		SPDLOG_DEBUG(
			"bpftime_get_stack only supports collect user stack!");
		return -ENOTSUP;
	}
	if (flags & BPF_F_USER_BUILD_ID) {
		SPDLOG_ERROR("bpftime_get_stack doesn't support buildid!");
		return -ENOTSUP;
	}
	auto *buffer = reinterpret_cast<void *>(static_cast<uintptr_t>(buf));
	if (size % sizeof(uint64_t) != 0) {
		SPDLOG_ERROR(
			"bpftime_get_stack buffer size must be a multiple of {} bytes",
			sizeof(uint64_t));
		if (size > 0)
			memset(buffer, 0, size);
		return -EINVAL;
	}
	const int ATTACH_UPROBE = 6;

	auto &attach_ctx = get_global_attach_ctx();
	auto attach_impl =
		attach_ctx.get_attach_impl_by_attach_type(ATTACH_UPROBE);
	if (!attach_impl.has_value()) {
		SPDLOG_ERROR(
			"Unable to get stack id: frida_uprobe_attach_impl not registered");
		return -ENOTSUP;
	}
	attach::stack_walk_data walk;
	auto raw_ptr = (*attach_impl)
			       ->call_attach_specific_function("generate_stack",
							       &walk);
	if (raw_ptr == nullptr || walk.count == 0) {
		// No frames: the probe ran where no userspace stack can be
		// unwound, such as a syscall-trace or tracepoint program
		// whose pt_regs was synthesized. The kernel reports that as
		// an empty stack, so copy nothing and report zero bytes
		// instead of an error the program would have to special-case.
		SPDLOG_DEBUG("No stack frames available");
		if (size > 0)
			memset(buffer, 0, size);
		return 0;
	}
	const uint64_t *frames = walk.frames;
	size_t frame_count = walk.count;

	auto frames_to_skip = static_cast<size_t>(flags & BPF_F_SKIP_FIELD_MASK);
	SPDLOG_DEBUG("Skipping {} frames", frames_to_skip);
	if (frames_to_skip > frame_count) {
		if (size > 0)
			memset(buffer, 0, size);
		return -EFAULT;
	}

	// Both operands must be the same type: on arm64 Darwin uint64_t is
	// unsigned long long while size_t is unsigned long, which libc++ rejects.
	auto frames_to_copy =
		std::min(static_cast<size_t>(frame_count - frames_to_skip),
			 static_cast<size_t>(size / sizeof(uint64_t)));
	auto size_to_copy = frames_to_copy * sizeof(uint64_t);
	SPDLOG_DEBUG("Copied {} bytes of stack", size_to_copy);
	if (size_to_copy > 0)
		memcpy(buffer, frames + frames_to_skip, size_to_copy);
	if (size > size_to_copy)
		memset(static_cast<char *>(buffer) + size_to_copy, 0,
		       size - size_to_copy);
	return static_cast<int64_t>(size_to_copy);
}

} // extern "C"

namespace bpftime
{

bool helper_is_async_signal_safe(unsigned index, void *fn)
{
	if ((index == 173 && fn == (void *)bpftime_get_func_ip) ||
	    (index == 174 && fn == (void *)bpftime_get_attach_cookie) ||
	    (index == 58 && fn == (void *)bpftime_override_return) ||
	    (index == 187 && fn == (void *)bpftime_set_retval))
		return true;
#if BPFTIME_HAVE_TRAP_UPROBE_ATTACH
	using namespace attach::trap;
	if ((index == 183 && fn == (void *)bpftime_trap_get_func_arg) ||
	    (index == 184 && fn == (void *)bpftime_trap_get_func_ret) ||
	    (index == 186 && fn == (void *)bpftime_trap_get_retval))
		return true;
#endif
	return false;
}

// copied from kernel/include/uapi/linux/bpf.h
enum bpf_func_id {
	BPF_FUNC_unspec = 0,
	BPF_FUNC_map_lookup_elem = 1,
	BPF_FUNC_map_update_elem = 2,
	BPF_FUNC_map_delete_elem = 3,
	BPF_FUNC_probe_read = 4,
	BPF_FUNC_ktime_get_ns = 5,
	BPF_FUNC_trace_printk = 6,
	BPF_FUNC_get_prandom_u32 = 7,
	BPF_FUNC_get_smp_processor_id = 8,
	BPF_FUNC_skb_store_bytes = 9,
	BPF_FUNC_l3_csum_replace = 10,
	BPF_FUNC_l4_csum_replace = 11,
	BPF_FUNC_tail_call = 12,
	BPF_FUNC_clone_redirect = 13,
	BPF_FUNC_get_current_pid_tgid = 14,
	BPF_FUNC_get_current_uid_gid = 15,
	BPF_FUNC_get_current_comm = 16,
	BPF_FUNC_get_cgroup_classid = 17,
	BPF_FUNC_skb_vlan_push = 18,
	BPF_FUNC_skb_vlan_pop = 19,
	BPF_FUNC_skb_get_tunnel_key = 20,
	BPF_FUNC_skb_set_tunnel_key = 21,
	BPF_FUNC_perf_event_read = 22,
	BPF_FUNC_redirect = 23,
	BPF_FUNC_get_route_realm = 24,
	BPF_FUNC_perf_event_output = 25,
	BPF_FUNC_skb_load_bytes = 26,
	BPF_FUNC_get_stackid = 27,
	BPF_FUNC_csum_diff = 28,
	BPF_FUNC_skb_get_tunnel_opt = 29,
	BPF_FUNC_skb_set_tunnel_opt = 30,
	BPF_FUNC_skb_change_proto = 31,
	BPF_FUNC_skb_change_type = 32,
	BPF_FUNC_skb_under_cgroup = 33,
	BPF_FUNC_get_hash_recalc = 34,
	BPF_FUNC_get_current_task = 35,
	BPF_FUNC_probe_write_user = 36,
	BPF_FUNC_current_task_under_cgroup = 37,
	BPF_FUNC_skb_change_tail = 38,
	BPF_FUNC_skb_pull_data = 39,
	BPF_FUNC_csum_update = 40,
	BPF_FUNC_set_hash_invalid = 41,
	BPF_FUNC_get_numa_node_id = 42,
	BPF_FUNC_skb_change_head = 43,
	BPF_FUNC_xdp_adjust_head = 44,
	BPF_FUNC_probe_read_str = 45,
	BPF_FUNC_get_socket_cookie = 46,
	BPF_FUNC_get_socket_uid = 47,
	BPF_FUNC_set_hash = 48,
	BPF_FUNC_setsockopt = 49,
	BPF_FUNC_skb_adjust_room = 50,
	BPF_FUNC_redirect_map = 51,
	BPF_FUNC_sk_redirect_map = 52,
	BPF_FUNC_sock_map_update = 53,
	BPF_FUNC_xdp_adjust_meta = 54,
	BPF_FUNC_perf_event_read_value = 55,
	BPF_FUNC_perf_prog_read_value = 56,
	BPF_FUNC_getsockopt = 57,
	BPF_FUNC_override_return = 58,
	BPF_FUNC_sock_ops_cb_flags_set = 59,
	BPF_FUNC_msg_redirect_map = 60,
	BPF_FUNC_msg_apply_bytes = 61,
	BPF_FUNC_msg_cork_bytes = 62,
	BPF_FUNC_msg_pull_data = 63,
	BPF_FUNC_bind = 64,
	BPF_FUNC_xdp_adjust_tail = 65,
	BPF_FUNC_skb_get_xfrm_state = 66,
	BPF_FUNC_get_stack = 67,
	BPF_FUNC_skb_load_bytes_relative = 68,
	BPF_FUNC_fib_lookup = 69,
	BPF_FUNC_sock_hash_update = 70,
	BPF_FUNC_msg_redirect_hash = 71,
	BPF_FUNC_sk_redirect_hash = 72,
	BPF_FUNC_lwt_push_encap = 73,
	BPF_FUNC_lwt_seg6_store_bytes = 74,
	BPF_FUNC_lwt_seg6_adjust_srh = 75,
	BPF_FUNC_lwt_seg6_action = 76,
	BPF_FUNC_rc_repeat = 77,
	BPF_FUNC_rc_keydown = 78,
	BPF_FUNC_skb_cgroup_id = 79,
	BPF_FUNC_get_current_cgroup_id = 80,
	BPF_FUNC_get_local_storage = 81,
	BPF_FUNC_sk_select_reuseport = 82,
	BPF_FUNC_skb_ancestor_cgroup_id = 83,
	BPF_FUNC_sk_lookup_tcp = 84,
	BPF_FUNC_sk_lookup_udp = 85,
	BPF_FUNC_sk_release = 86,
	BPF_FUNC_map_push_elem = 87,
	BPF_FUNC_map_pop_elem = 88,
	BPF_FUNC_map_peek_elem = 89,
	BPF_FUNC_msg_push_data = 90,
	BPF_FUNC_msg_pop_data = 91,
	BPF_FUNC_rc_pointer_rel = 92,
	BPF_FUNC_spin_lock = 93,
	BPF_FUNC_spin_unlock = 94,
	BPF_FUNC_sk_fullsock = 95,
	BPF_FUNC_tcp_sock = 96,
	BPF_FUNC_skb_ecn_set_ce = 97,
	BPF_FUNC_get_listener_sock = 98,
	BPF_FUNC_skc_lookup_tcp = 99,
	BPF_FUNC_tcp_check_syncookie = 100,
	BPF_FUNC_sysctl_get_name = 101,
	BPF_FUNC_sysctl_get_current_value = 102,
	BPF_FUNC_sysctl_get_new_value = 103,
	BPF_FUNC_sysctl_set_new_value = 104,
	BPF_FUNC_strtol = 105,
	BPF_FUNC_strtoul = 106,
	BPF_FUNC_sk_storage_get = 107,
	BPF_FUNC_sk_storage_delete = 108,
	BPF_FUNC_send_signal = 109,
	BPF_FUNC_tcp_gen_syncookie = 110,
	BPF_FUNC_skb_output = 111,
	BPF_FUNC_probe_read_user = 112,
	BPF_FUNC_probe_read_kernel = 113,
	BPF_FUNC_probe_read_user_str = 114,
	BPF_FUNC_probe_read_kernel_str = 115,
	BPF_FUNC_tcp_send_ack = 116,
	BPF_FUNC_send_signal_thread = 117,
	BPF_FUNC_jiffies64 = 118,
	BPF_FUNC_read_branch_records = 119,
	BPF_FUNC_get_ns_current_pid_tgid = 120,
	BPF_FUNC_xdp_output = 121,
	BPF_FUNC_get_netns_cookie = 122,
	BPF_FUNC_get_current_ancestor_cgroup_id = 123,
	BPF_FUNC_sk_assign = 124,
	BPF_FUNC_ktime_get_boot_ns = 125,
	BPF_FUNC_seq_printf = 126,
	BPF_FUNC_seq_write = 127,
	BPF_FUNC_sk_cgroup_id = 128,
	BPF_FUNC_sk_ancestor_cgroup_id = 129,
	BPF_FUNC_ringbuf_output = 130,
	BPF_FUNC_ringbuf_reserve = 131,
	BPF_FUNC_ringbuf_submit = 132,
	BPF_FUNC_ringbuf_discard = 133,
	BPF_FUNC_ringbuf_query = 134,
	BPF_FUNC_csum_level = 135,
	BPF_FUNC_skc_to_tcp6_sock = 136,
	BPF_FUNC_skc_to_tcp_sock = 137,
	BPF_FUNC_skc_to_tcp_timewait_sock = 138,
	BPF_FUNC_skc_to_tcp_request_sock = 139,
	BPF_FUNC_skc_to_udp6_sock = 140,
	BPF_FUNC_get_task_stack = 141,
	BPF_FUNC_load_hdr_opt = 142,
	BPF_FUNC_store_hdr_opt = 143,
	BPF_FUNC_reserve_hdr_opt = 144,
	BPF_FUNC_inode_storage_get = 145,
	BPF_FUNC_inode_storage_delete = 146,
	BPF_FUNC_d_path = 147,
	BPF_FUNC_copy_from_user = 148,
	BPF_FUNC_snprintf_btf = 149,
	BPF_FUNC_seq_printf_btf = 150,
	BPF_FUNC_skb_cgroup_classid = 151,
	BPF_FUNC_redirect_neigh = 152,
	BPF_FUNC_per_cpu_ptr = 153,
	BPF_FUNC_this_cpu_ptr = 154,
	BPF_FUNC_redirect_peer = 155,
	BPF_FUNC_task_storage_get = 156,
	BPF_FUNC_task_storage_delete = 157,
	BPF_FUNC_get_current_task_btf = 158,
	BPF_FUNC_bprm_opts_set = 159,
	BPF_FUNC_ktime_get_coarse_ns = 160,
	BPF_FUNC_ima_inode_hash = 161,
	BPF_FUNC_sock_from_file = 162,
	BPF_FUNC_check_mtu = 163,
	BPF_FUNC_for_each_map_elem = 164,
	BPF_FUNC_snprintf = 165,
	BPF_FUNC_sys_bpf = 166,
	BPF_FUNC_btf_find_by_name_kind = 167,
	BPF_FUNC_sys_close = 168,
	BPF_FUNC_timer_init = 169,
	BPF_FUNC_timer_set_callback = 170,
	BPF_FUNC_timer_start = 171,
	BPF_FUNC_timer_cancel = 172,
	BPF_FUNC_get_func_ip = 173,
	BPF_FUNC_get_attach_cookie = 174,
	BPF_FUNC_task_pt_regs = 175,
	BPF_FUNC_get_branch_snapshot = 176,
	BPF_FUNC_trace_vprintk = 177,
	BPF_FUNC_skc_to_unix_sock = 178,
	BPF_FUNC_kallsyms_lookup_name = 179,
	BPF_FUNC_find_vma = 180,
	BPF_FUNC_loop = 181,
	BPF_FUNC_strncmp = 182,
	BPF_FUNC_get_func_arg = 183,
	BPF_FUNC_get_func_ret = 184,
	BPF_FUNC_get_func_arg_cnt = 185,
	BPF_FUNC_get_retval = 186,
	BPF_FUNC_set_retval = 187,
	BPF_FUNC_xdp_get_buff_len = 188,
	BPF_FUNC_xdp_load_bytes = 189,
	BPF_FUNC_xdp_store_bytes = 190,
	BPF_FUNC_copy_from_user_task = 191,
	BPF_FUNC_skb_set_tstamp = 192,
	BPF_FUNC_ima_file_hash = 193,
	BPF_FUNC_kptr_xchg = 194,
	BPF_FUNC_map_lookup_percpu_elem = 195,
	BPF_FUNC_skc_to_mptcp_sock = 196,
	BPF_FUNC_dynptr_from_mem = 197,
	BPF_FUNC_ringbuf_reserve_dynptr = 198,
	BPF_FUNC_ringbuf_submit_dynptr = 199,
	BPF_FUNC_ringbuf_discard_dynptr = 200,
	BPF_FUNC_dynptr_read = 201,
	BPF_FUNC_dynptr_write = 202,
	BPF_FUNC_dynptr_data = 203,
	BPF_FUNC_tcp_raw_gen_syncookie_ipv4 = 204,
	BPF_FUNC_tcp_raw_gen_syncookie_ipv6 = 205,
	BPF_FUNC_tcp_raw_check_syncookie_ipv4 = 206,
	BPF_FUNC_tcp_raw_check_syncookie_ipv6 = 207,
	BPF_FUNC_ktime_get_tai_ns = 208,
	BPF_FUNC_user_ringbuf_drain = 209,
	BPF_FUNC_cgrp_storage_get = 210,
	BPF_FUNC_cgrp_storage_delete = 211,
	__BPF_FUNC_MAX_ID = 212,
};

int bpftime_helper_group::register_helper(const bpftime_helper_info &info)
{
	if (info.index > 999) {
		SPDLOG_ERROR("Helper id should be 0-999, found {}", info.index);
		return -1;
	}
	if (helper_map.find(info.index) == helper_map.end()) {
		helper_map[info.index] = info;
	} else {
		// found the same helper id
		if (helper_map[info.index].fn != info.fn) {
			SPDLOG_ERROR("Helper id already exists for {}",
				     info.name);
			return -1;
		}
		// else, ignore the same helper
	}
	return 0;
}

int bpftime_helper_group::add_helper_group_to_prog(bpftime_prog *prog) const
{
	for (auto it : helper_map) {
		prog->bpftime_prog_register_raw_helper(it.second);
	}
	return 0;
}

std::vector<int32_t> bpftime_helper_group::get_helper_ids() const
{
	std::vector<int32_t> result;
	for (const auto &[k, _] : this->helper_map) {
		result.push_back(k);
	}
	return result;
}

// Utility function to get the UFUNC helper group
const bpftime_helper_group &bpftime_helper_group::get_ufunc_helper_group()
{
	const bpftime_helper_group &get_extension_helper_group();

	return get_extension_helper_group();
}

// Utility function to get the kernel utilities helper group
const bpftime_helper_group &
bpftime_helper_group::get_kernel_utils_helper_group()
{
	static const bpftime_helper_group kernel_helper_group = {
		{ { BPF_FUNC_probe_read,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_read,
			    .name = "bpf_probe_read",
			    .fn = (void *)bpftime_probe_read,
		    } },
		  { BPF_FUNC_get_smp_processor_id,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_smp_processor_id,
			    .name = "bpf_get_smp_processor_id",
			    .fn = (void *)bpftime_get_smp_processor_id,
		    } },
		  { BPF_FUNC_csum_diff,
		    bpftime_helper_info{
			    .index = BPF_FUNC_csum_diff,
			    .name = "bpf_csum_diff",
			    .fn = (void *)bpftime_csum_diff,
		    } },
		  { BPF_FUNC_xdp_adjust_head,
		    bpftime_helper_info{
			    .index = BPF_FUNC_xdp_adjust_head,
			    .name = "bpf_xdp_adjust_head",
			    .fn = (void *)bpftime_xdp_adjust_head,
		    } },
		  { BPF_FUNC_xdp_adjust_tail,
		    bpftime_helper_info{
			    .index = BPF_FUNC_xdp_adjust_tail,
			    .name = "bpf_xdp_adjust_tail",
			    .fn = (void *)bpftime_xdp_adjust_tail,
		    } },
		  { BPF_FUNC_probe_read_kernel,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_read_kernel,
			    .name = "bpf_probe_read_kernel",
			    .fn = (void *)bpftime_probe_read,
		    } },
		  { BPF_FUNC_probe_read_user,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_read_user,
			    .name = "bpf_probe_read_user",
			    .fn = (void *)bpftime_probe_read,
		    } },
		  { BPF_FUNC_ktime_get_ns,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ktime_get_ns,
			    .name = "bpf_ktime_get_ns",
			    .fn = (void *)bpftime_ktime_get_ns,
		    } },
		  { BPF_FUNC_ktime_get_boot_ns,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ktime_get_boot_ns,
			    .name = "bpf_ktime_get_boot_ns",
			    .fn = (void *)bpftime_ktime_get_boot_ns,
		    } },
		  { BPF_FUNC_seq_write,
		    bpftime_helper_info{
			    .index = BPF_FUNC_seq_write,
			    .name = "bpf_seq_write",
			    .fn = (void *)bpftime_seq_write,
		    } },
		  { BPF_FUNC_trace_printk,
		    bpftime_helper_info{
			    .index = BPF_FUNC_trace_printk,
			    .name = "bpf_trace_printk",
			    .fn = (void *)bpftime_trace_printk,
		    } },
		  { BPF_FUNC_get_prandom_u32,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_prandom_u32,
			    .name = "bpf_get_prandom_u32",
			    .fn = (void *)bpftime_get_prandom_u32,
		    } },
		  { BPF_FUNC_get_current_pid_tgid,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_current_pid_tgid,
			    .name = "bpf_get_current_pid_tgid",
			    .fn = (void *)bpftime_get_current_pid_tgid,
		    } },
		  { BPF_FUNC_get_current_uid_gid,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_current_uid_gid,
			    .name = "bpf_get_current_uid_gid",
			    .fn = (void *)bpf_get_current_uid_gid } },
		  { BPF_FUNC_get_current_task,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_current_task,
			    .name = "bpf_get_current_task",
			    .fn = (void *)bpftime_get_current_task } },
		  { BPF_FUNC_get_current_task_btf,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_current_task_btf,
			    .name = "bpf_get_current_task_btf",
			    .fn = (void *)bpftime_get_current_task } },
		  { BPF_FUNC_get_current_comm,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_current_comm,
			    .name = "bpf_get_current_comm",
			    .fn = (void *)bpftime_get_current_comm,
		    } },
		  { BPF_FUNC_override_return,
		    bpftime_helper_info{
			    .index = BPF_FUNC_override_return,
			    .name = "bpf_override_return",
			    .fn = (void *)bpftime_override_return,
		    } },
		  { BPF_FUNC_strncmp,
		    bpftime_helper_info{
			    .index = BPF_FUNC_strncmp,
			    .name = "bpf_strncmp",
			    .fn = (void *)bpftime_strncmp,
		    } },
		  { BPF_FUNC_probe_write_user,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_write_user,
			    .name = "bpf_probe_write_user",
			    .fn = (void *)bpftime_probe_write_user,
		    } },
		  { BPF_FUNC_set_retval,
		    bpftime_helper_info{
			    .index = BPF_FUNC_set_retval,
			    .name = "bpf_set_retval",
			    .fn = (void *)bpftime_set_retval,
		    } },
		  { BPF_FUNC_probe_read_user_str,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_read_user_str,
			    .name = "bpf_probe_read_str",
			    .fn = (void *)bpf_probe_read_str,
		    } },
		  { BPF_FUNC_probe_read_str,
		    bpftime_helper_info{
			    .index = BPF_FUNC_probe_read_str,
			    .name = "bpf_probe_str",
			    .fn = (void *)bpf_probe_read_str,
		    } },
		  { BPF_FUNC_get_stack,
		    bpftime_helper_info{ .index = BPF_FUNC_get_stack,
					 .name = "bpf_get_stack",
					 .fn = (void *)bpftime_get_stack } },
		  { BPF_FUNC_get_stackid,
		    bpftime_helper_info{ .index = BPF_FUNC_get_stackid,
					 .name = "bpf_get_stackid",
					 .fn = (void *)bpftime_get_stackid } },

		  { BPF_FUNC_ktime_get_coarse_ns,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ktime_get_coarse_ns,
			    .name = "bpf_ktime_get_coarse_ns",
			    .fn = (void *)bpf_ktime_get_coarse_ns } },

		  { BPF_FUNC_ringbuf_reserve,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ringbuf_reserve,
			    .name = "bpf_ringbuf_reserve",
			    .fn = (void *)bpf_ringbuf_reserve,
		    } },
		  { BPF_FUNC_ringbuf_submit,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ringbuf_submit,
			    .name = "bpf_ringbuf_submit",
			    .fn = (void *)bpf_ringbuf_submit,
		    } },
		  { BPF_FUNC_ringbuf_discard,
		    bpftime_helper_info{
			    .index = BPF_FUNC_ringbuf_discard,
			    .name = "bpf_ringbuf_discard",
			    .fn = (void *)bpf_ringbuf_discard,
		    } },
		  { BPF_FUNC_perf_event_output,
		    bpftime_helper_info{ .index = BPF_FUNC_perf_event_output,
					 .name = "bpf_perf_event_output",
					 .fn = (void *)bpf_perf_event_output } },
		  { BPF_FUNC_ringbuf_output,
		    bpftime_helper_info{ .index = BPF_FUNC_ringbuf_output,
					 .name = "bpf_ringbuf_output",
					 .fn = (void *)bpf_ringbuf_output } },
		  { BPF_FUNC_tail_call,
		    bpftime_helper_info{ .index = BPF_FUNC_tail_call,
					 .name = "bpf_tail_call",
					 .fn = (void *)bpftime_tail_call } },
		  { BPF_FUNC_get_attach_cookie,
		    bpftime_helper_info{
			    .index = BPF_FUNC_get_attach_cookie,
			    .name = "bpf_get_attach_cookie",
			    .fn = (void *)bpftime_get_attach_cookie } },
		  { BPF_FUNC_get_func_ip,
		    bpftime_helper_info{ .index = BPF_FUNC_get_func_ip,
					 .name = "bpf_get_func_ip",
					 .fn = (void *)bpftime_get_func_ip } } }

	};

	return kernel_helper_group;
}
const bpftime_helper_group &bpftime_helper_group::get_shm_maps_helper_group()
{
	static const bpftime_helper_group shm_maps_group = { {
		{ BPF_FUNC_map_lookup_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_lookup_elem,
			  .name = "bpf_map_lookup_elem",
			  .fn = (void *)bpftime_map_lookup_elem_helper,
		  } },
		{ BPF_FUNC_map_update_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_update_elem,
			  .name = "bpf_map_update_elem",
			  .fn = (void *)bpftime_map_update_elem_helper,
		  } },
		{ BPF_FUNC_map_delete_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_delete_elem,
			  .name = "bpf_map_delete_elem",
			  .fn = (void *)bpftime_map_delete_elem_helper,
		  } },
		{ BPF_FUNC_map_push_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_push_elem,
			  .name = "bpf_map_push_elem",
			  .fn = (void *)bpftime_map_push_elem_helper,
		  } },
		{ BPF_FUNC_map_pop_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_pop_elem,
			  .name = "bpf_map_pop_elem",
			  .fn = (void *)bpftime_map_pop_elem_helper,
		  } },
		{ BPF_FUNC_map_peek_elem,
		  bpftime_helper_info{
			  .index = BPF_FUNC_map_peek_elem,
			  .name = "bpf_map_peek_elem",
			  .fn = (void *)bpftime_map_peek_elem_helper,
		  } },
	} };

	return shm_maps_group;
}

#ifdef ENABLE_BPFTIME_VERIFIER
std::map<int32_t, verifier::BpftimeHelperProrotype> get_ufunc_helper_protos()
{
	using namespace verifier;
	std::map<int32_t, BpftimeHelperProrotype> result;
	result[UFUNC_HELPER_ID_FIND_ID] = BpftimeHelperProrotype{
		.name = "__ebpf_call_find_ufunc_id",
		.return_type = EBPF_RETURN_TYPE_INTEGER,
		.argument_type = { EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM_OR_NULL,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE }
	};

	result[UFUNC_HELPER_ID_DISPATCHER] = BpftimeHelperProrotype{
		.name = "__ebpf_call_ufunc_dispatcher",
		.return_type = EBPF_RETURN_TYPE_INTEGER,
		.argument_type = { EBPF_ARGUMENT_TYPE_ANYTHING,
				   EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM_OR_NULL,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE }
	};

	return result;
}
#endif

} // namespace bpftime
