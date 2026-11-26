/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#ifndef ROBUST_SHM_MUTEX_HPP
#define ROBUST_SHM_MUTEX_HPP

#include "spdlog/spdlog.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <signal.h>
#endif

namespace bpftime
{
namespace detail
{

inline constexpr uint64_t SHM_MUTEX_TAINTED = uint64_t{ 1 } << 63;

inline void shm_cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	asm volatile("yield" ::: "memory");
#else
	std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

#if defined(__linux__)
inline char *append_str(char *dst, const char *s) noexcept
{
	while (*s)
		*dst++ = *s++;
	return dst;
}

inline char *append_u32(char *dst, uint32_t v) noexcept
{
	char tmp[16];
	int n = 0;
	do {
		tmp[n++] = static_cast<char>('0' + (v % 10));
		v /= 10;
	} while (v != 0);
	while (n > 0)
		*dst++ = tmp[--n];
	return dst;
}

// Allocation-free reader for /proc/<pid>/task/<tid>/stat state character.
// Returns -1 if the thread/process entry no longer exists, 0 if unreadable,
// or 1 with state_out populated.
inline int read_proc_thread_state(uint32_t pid, uint32_t tid,
				  char &state_out) noexcept
{
	char path[64];
	char *end = append_str(path, "/proc/");
	end = append_u32(end, pid);
	end = append_str(end, "/task/");
	end = append_u32(end, tid);
	end = append_str(end, "/stat");
	*end = '\0';

	int fd = ::open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return (errno == ENOENT || errno == ESRCH) ? -1 : 0;
	char buf[256];
	ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
	::close(fd);
	if (n <= 0)
		return 0;
	buf[n] = '\0';

	const char *rparen = std::strrchr(buf, ')');
	if (!rparen || rparen[1] != ' ' || rparen[2] == '\0')
		return 0;
	state_out = rparen[2];
	return 1;
}
#endif

// Static-TLS (initial-exec on ELF) so dlopen'd agent libraries never trigger
// __tls_get_addr heap allocation on the hot path.
inline thread_local uint64_t shm_mutex_owner_token
#if defined(__linux__)
	__attribute__((tls_model("initial-exec")))
#endif
	= 0;

inline std::atomic<uint32_t> shm_mutex_pidns_tag{ 0 };
inline std::atomic<bool> shm_mutex_atfork_registered{ false };

// Async-signal-safe query of the current process's PID-namespace inode tag
// (non-zero 19-bit value; 0 means unknown/unavailable).
inline uint32_t query_self_pidns_tag() noexcept
{
#if defined(__linux__)
	struct stat st{};
	if (::stat("/proc/self/ns/pid", &st) == 0 && st.st_ino != 0) {
		const uint64_t ino = static_cast<uint64_t>(st.st_ino);
		const uint32_t tag =
			static_cast<uint32_t>((ino ^ (ino >> 19)) & 0x7FFFFu);
		return tag == 0 ? 1u : tag;
	}
#endif
	return 0;
}

inline void reset_shm_mutex_owner_token_in_child() noexcept
{
	shm_mutex_owner_token = 0;
	shm_mutex_pidns_tag.store(query_self_pidns_tag(),
				  std::memory_order_relaxed);
}

inline void ensure_shm_mutex_process_init() noexcept
{
	if (shm_mutex_atfork_registered.load(std::memory_order_acquire))
		return;
	const int saved_errno = errno;
	const uint32_t tag = query_self_pidns_tag();
	if (tag != 0)
		shm_mutex_pidns_tag.store(tag, std::memory_order_relaxed);
	if (!shm_mutex_atfork_registered.exchange(true,
						  std::memory_order_acq_rel)) {
		(void)pthread_atfork(nullptr, nullptr,
				     reset_shm_mutex_owner_token_in_child);
	}
	errno = saved_errno;
}

namespace
{
__attribute__((constructor)) void init_shm_mutex_module() noexcept
{
	ensure_shm_mutex_process_init();
}
} // namespace

// Async-signal-safe per-thread owner token initializer: uses only raw
// SYS_getpid / SYS_gettid syscalls once the module constructor has cached
// shm_mutex_pidns_tag, so first-lock inside a trap-uprobe signal handler
// never calls pthread_atfork, malloc, or open/read/close.
__attribute__((noinline)) inline uint64_t init_shm_owner_token_slow() noexcept
{
	const int saved_errno = errno;
	uint64_t token = 1;
#if defined(__linux__)
	uint32_t ns_tag = shm_mutex_pidns_tag.load(std::memory_order_relaxed);
	if (__builtin_expect(ns_tag == 0, 0)) {
		ns_tag = query_self_pidns_tag();
		if (ns_tag != 0)
			shm_mutex_pidns_tag.store(ns_tag,
						  std::memory_order_relaxed);
	}
	const uint32_t pid =
		static_cast<uint32_t>(::syscall(SYS_getpid)) & 0x3FFFFFu;
	const uint32_t tid =
		static_cast<uint32_t>(::syscall(SYS_gettid)) & 0x3FFFFFu;
	token = (static_cast<uint64_t>(ns_tag & 0x7FFFFu) << 44) |
		(static_cast<uint64_t>(pid) << 22) |
		static_cast<uint64_t>(tid ? tid : 1u);
#elif defined(__APPLE__)
	uint64_t tid64 = 0;
	pthread_threadid_np(nullptr, &tid64);
	const uint32_t pid = static_cast<uint32_t>(::getpid()) & 0x3FFFFFu;
	const uint32_t tid = static_cast<uint32_t>(tid64) & 0x3FFFFFu;
	token = (static_cast<uint64_t>(pid) << 22) |
		static_cast<uint64_t>(tid ? tid : 1u);
#endif
	shm_mutex_owner_token = token;
	errno = saved_errno;
	return token;
}

inline uint64_t current_shm_owner_token() noexcept
{
	uint64_t t = shm_mutex_owner_token;
	if (__builtin_expect(t != 0, 1))
		return t;
	return init_shm_owner_token_slow();
}

inline bool shm_owner_is_dead(uint64_t owner) noexcept
{
	if (owner == 0 || (owner & SHM_MUTEX_TAINTED) != 0)
		return false;
#if defined(__linux__)
	const uint32_t tid = static_cast<uint32_t>(owner & 0x3FFFFFu);
	const uint32_t pid = static_cast<uint32_t>((owner >> 22) & 0x3FFFFFu);
	const uint32_t owner_ns_tag =
		static_cast<uint32_t>((owner >> 44) & 0x7FFFFu);
	if (pid == 0 || tid == 0 || owner_ns_tag == 0)
		return false;
	uint32_t waiter_ns_tag =
		shm_mutex_pidns_tag.load(std::memory_order_relaxed);
	if (waiter_ns_tag == 0)
		waiter_ns_tag = query_self_pidns_tag();
	// Never interpret pid/tid from another (or unknown) PID namespace:
	// treat cross-namespace holders as alive so waiters keep waiting.
	if (waiter_ns_tag == 0 || owner_ns_tag != waiter_ns_tag)
		return false;
	if (::syscall(SYS_tgkill, static_cast<int>(pid), static_cast<int>(tid),
		      0) != 0 &&
	    errno == ESRCH)
		return true;
	char state = 0;
	int rc = read_proc_thread_state(pid, tid, state);
	if (rc <= 0) {
		// procfs can be missing or unreadable while the holder thread
		// is still alive (thread-group leader exit, hidepid mounts,
		// permission/fd errors). Since the first tgkill did not confirm
		// thread exit, require a second tgkill returning ESRCH before
		// treating the holder as dead; keep live or EPERM holders safe.
		return ::syscall(SYS_tgkill, static_cast<int>(pid),
				 static_cast<int>(tid), 0) != 0 &&
		       errno == ESRCH;
	}
	if (rc > 0 && (state == 'Z' || state == 'X' || state == 'x'))
		return true;
	return false;
#elif defined(__APPLE__)
	const uint32_t pid = static_cast<uint32_t>((owner >> 22) & 0x3FFFFFu);
	if (pid == 0)
		return false;
	return ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
#else
	return false;
#endif
}

} // namespace detail

// Process-shared, 8-byte atomic spinlock with same-PID-namespace dead-holder
// detection on the slow path:
//   [63: tainted | 62..44: 19-bit pidns inode tag | 43..22: pid | 21..0: tid]
// Fast path is 1 initial-exec TLS read + 1 acquire CAS (0 -> token) and 1
// release store (0). Cross-PID-namespace holders do not match waiter_ns_tag
// and stay conservative (wait until unlock without false taint or takeover).
class robust_shm_mutex {
	static_assert(std::atomic<uint64_t>::is_always_lock_free,
		      "robust_shm_mutex requires lock-free 64-bit atomics");
	std::atomic<uint64_t> state{ 0 };

	__attribute__((noinline)) bool
	lock_slow(uint64_t me, bool recover, const char *label) noexcept
	{
		const int saved_errno = errno;
		uint64_t tracked_owner = 0;
		uint32_t same_owner_yields = 0;

		for (;;) {
			for (int spin = 0; spin < 128; ++spin) {
				uint64_t cur =
					state.load(std::memory_order_relaxed);
				if (cur == 0) {
					if (state.compare_exchange_weak(
						    cur, me,
						    std::memory_order_acquire,
						    std::memory_order_relaxed)) {
						errno = saved_errno;
						return true;
					}
					continue;
				}
				if (!recover &&
				    (cur & detail::SHM_MUTEX_TAINTED) != 0) {
					errno = EBUSY;
					return false;
				}
				detail::shm_cpu_relax();
			}

			uint64_t cur = state.load(std::memory_order_acquire);
			if (cur == 0)
				continue;
			if (!recover &&
			    (cur & detail::SHM_MUTEX_TAINTED) != 0) {
				errno = EBUSY;
				return false;
			}
			if (cur != tracked_owner) {
				tracked_owner = cur;
				same_owner_yields = 0;
			}
			if (++same_owner_yields < 512) {
				sched_yield();
				continue;
			}

			if (detail::shm_owner_is_dead(cur)) {
				if (recover) {
					if (state.compare_exchange_strong(
						    cur, me,
						    std::memory_order_acquire,
						    std::memory_order_relaxed)) {
						errno = saved_errno;
						return true;
					}
					continue;
				}
				if (state.compare_exchange_strong(
					    cur, detail::SHM_MUTEX_TAINTED,
					    std::memory_order_acq_rel,
					    std::memory_order_relaxed)) {
					try {
						SPDLOG_ERROR(
							"SHM robust lock (guard '{}') owner died; latching tainted state, future lockers fail with EBUSY",
							label ? label :
								"unlabeled");
					} catch (...) {
					}
				}
				errno = EBUSY;
				return false;
			}

			timespec ts{ 0, 100000 }; // 100 us
			nanosleep(&ts, nullptr);
		}
	}

    public:
	constexpr robust_shm_mutex() noexcept = default;
	~robust_shm_mutex() = default;
	robust_shm_mutex(const robust_shm_mutex &) = delete;
	robust_shm_mutex &operator=(const robust_shm_mutex &) = delete;

	bool lock(const char *label = nullptr) noexcept
	{
		uint64_t me = detail::current_shm_owner_token();
		uint64_t expected = 0;
		if (__builtin_expect(
			    state.compare_exchange_strong(
				    expected, me, std::memory_order_acquire,
				    std::memory_order_relaxed),
			    1))
			return true;
		return lock_slow(me, false, label);
	}

	bool lock_and_recover() noexcept
	{
		uint64_t me = detail::current_shm_owner_token();
		uint64_t expected = 0;
		if (__builtin_expect(
			    state.compare_exchange_strong(
				    expected, me, std::memory_order_acquire,
				    std::memory_order_relaxed),
			    1))
			return true;
		return lock_slow(me, true, nullptr);
	}

	void unlock() noexcept
	{
		state.store(0, std::memory_order_release);
	}
};

class robust_shm_lock_guard {
	robust_shm_mutex *mutex;

    public:
	explicit robust_shm_lock_guard(robust_shm_mutex &m,
				       const char *label = nullptr) noexcept
		: mutex(m.lock(label) ? &m : nullptr)
	{
	}
	~robust_shm_lock_guard()
	{
		if (mutex)
			mutex->unlock();
	}
	bool owns_lock() const noexcept { return mutex != nullptr; }
	robust_shm_lock_guard(const robust_shm_lock_guard &) = delete;
	robust_shm_lock_guard &operator=(const robust_shm_lock_guard &) = delete;
};

} // namespace bpftime

#endif // ROBUST_SHM_MUTEX_HPP
