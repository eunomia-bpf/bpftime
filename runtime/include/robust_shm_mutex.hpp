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
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace bpftime
{

// Process-shared pthread mutex for cross-process SHM synchronization (ROBUST on
// Linux): lock()/try_lock() latch ENOTRECOVERABLE on EOWNERDEAD; lock_and_recover()
// marks pure mutual-exclusion slots consistent.
class robust_shm_mutex {
	pthread_mutex_t mutex;
	std::atomic<uint32_t> unrecoverable{ 0 };
	const bool recover_dead_owner;

	void latch_dead_owner() noexcept
	{
		unrecoverable.store(1, std::memory_order_release);
#ifdef __linux__
		(void)pthread_mutex_consistent(&mutex);
#endif
		pthread_mutex_unlock(&mutex);
	}

    public:
	robust_shm_mutex(bool recover = false)
		: recover_dead_owner(recover)
	{
		pthread_mutexattr_t attr;
		int rc = pthread_mutexattr_init(&attr);
		if (rc != 0)
			throw std::runtime_error(
				"pthread_mutexattr_init failed: " +
				std::to_string(rc));
		if ((rc = pthread_mutexattr_setpshared(
			     &attr, PTHREAD_PROCESS_SHARED)) != 0
#ifdef __linux__
		    || (rc = pthread_mutexattr_setrobust(
				&attr, PTHREAD_MUTEX_ROBUST)) != 0
#endif
		    || (rc = pthread_mutex_init(&mutex, &attr)) != 0) {
			pthread_mutexattr_destroy(&attr);
			throw std::runtime_error(
				"failed to initialize process-shared robust mutex: " +
				std::to_string(rc));
		}
		pthread_mutexattr_destroy(&attr);
	}
	~robust_shm_mutex() { pthread_mutex_destroy(&mutex); }
	robust_shm_mutex(const robust_shm_mutex &) = delete;
	robust_shm_mutex &operator=(const robust_shm_mutex &) = delete;

	bool lock([[maybe_unused]] const char *label = nullptr)
	{
		if (unrecoverable.load(std::memory_order_acquire))
			return errno = EBUSY, false;
		int rc = pthread_mutex_lock(&mutex);
#ifdef __linux__
		if (rc == EOWNERDEAD) {
			if (recover_dead_owner &&
			    pthread_mutex_consistent(&mutex) == 0)
				return true;
			latch_dead_owner();
			try {
				SPDLOG_ERROR(
					"SHM robust mutex (guard '{}') owner died (EOWNERDEAD); latching unrecoverable, future lockers fail with EBUSY",
					label ? label : "unlabeled");
			} catch (...) {
			}
			return errno = EBUSY, false;
		}
		if (rc == ENOTRECOVERABLE)
			return errno = EBUSY, false;
#endif
		if (rc != 0)
			return errno = rc, false;
		if (unrecoverable.load(std::memory_order_acquire))
			return pthread_mutex_unlock(&mutex), errno = EBUSY, false;
		return true;
	}

	bool lock_and_recover()
	{
		int rc = pthread_mutex_lock(&mutex);
#ifdef __linux__
		if (rc == EOWNERDEAD) {
			rc = pthread_mutex_consistent(&mutex);
			if (rc != 0) {
				pthread_mutex_unlock(&mutex);
				errno = rc;
				return false;
			}
			return true;
		}
#endif
		if (rc != 0) {
			errno = rc;
			return false;
		}
		return true;
	}

	bool try_lock() noexcept
	{
		if (unrecoverable.load(std::memory_order_acquire))
			return false;
		const int saved = errno, rc = pthread_mutex_trylock(&mutex);
#ifdef __linux__
		if (rc == EOWNERDEAD) {
			if (recover_dead_owner &&
			    pthread_mutex_consistent(&mutex) == 0)
				return errno = saved, true;
			latch_dead_owner();
		}
#endif
		bool ok = (rc == 0);
		if (ok && unrecoverable.load(std::memory_order_acquire)) {
			pthread_mutex_unlock(&mutex);
			ok = false;
		}
		errno = saved;
		return ok;
	}

	void unlock() { pthread_mutex_unlock(&mutex); }
};

// L1 for structures a dead owner cannot leave inconsistent: lock()/try_lock()
// mark the mutex consistent on EOWNERDEAD and continue instead of latching.
struct robust_shm_recovering_mutex : robust_shm_mutex {
	robust_shm_recovering_mutex() : robust_shm_mutex(true) {}
};

inline std::atomic<uint64_t> shm_trylock_skips{ 0 };
inline uint64_t get_shm_trylock_skip_count() noexcept
{
	return shm_trylock_skips.load(std::memory_order_relaxed);
}

inline int try_lease_shm_slot(robust_shm_recovering_mutex *slots, int count,
			      int preferred) noexcept
{
	if (!slots || count <= 0)
		return -1;
	const int start = preferred >= 0 ? (preferred % count) : 0;
	for (int i = 0; i < count; ++i) {
		const int idx = (start + i) % count;
		if (slots[idx].try_lock())
			return idx;
	}
	shm_trylock_skips.fetch_add(1, std::memory_order_relaxed);
	return -1;
}

inline void release_shm_slot(robust_shm_recovering_mutex *slots, int count,
			     int slot) noexcept
{
	if (slots && slot >= 0 && slot < count)
		slots[slot].unlock();
}

class robust_shm_lock_guard {
	robust_shm_mutex *mutex;

    public:
	explicit robust_shm_lock_guard(robust_shm_mutex &m,
				       const char *label = nullptr)
		: mutex(m.lock(label) ? &m : nullptr)
	{
	}
	~robust_shm_lock_guard()
	{
		if (mutex)
			mutex->unlock();
	}
	bool owns_lock() const { return mutex != nullptr; }
	robust_shm_lock_guard(const robust_shm_lock_guard &) = delete;
	robust_shm_lock_guard &operator=(const robust_shm_lock_guard &) = delete;
};

class robust_shm_trylock_guard {
	robust_shm_mutex *mutex;

    public:
	explicit robust_shm_trylock_guard(robust_shm_mutex &m) noexcept
		: mutex(m.try_lock() ? &m : nullptr)
	{
		if (!mutex)
			shm_trylock_skips.fetch_add(1,
						    std::memory_order_relaxed);
	}
	~robust_shm_trylock_guard()
	{
		if (mutex)
			mutex->unlock();
	}
	bool owns_lock() const noexcept { return mutex != nullptr; }
	robust_shm_trylock_guard(const robust_shm_trylock_guard &) = delete;
	robust_shm_trylock_guard &
	operator=(const robust_shm_trylock_guard &) = delete;
};

template <typename Fn>
inline decltype(auto)
with_robust_shm_lock(bool should_lock, bool blocking, robust_shm_mutex &lock,
		     Fn &&fn, const char *label = "bpf_map_handler")
{
	using Ret = std::invoke_result_t<Fn>;
	if (!should_lock)
		return fn();
	if (blocking) {
		robust_shm_lock_guard g(lock, label);
		if (!g.owns_lock()) {
			if constexpr (std::is_pointer_v<Ret>)
				return Ret(nullptr);
			return Ret(-1);
		}
		return fn();
	}
	robust_shm_trylock_guard g(lock);
	if (!g.owns_lock()) {
		if constexpr (std::is_pointer_v<Ret>)
			return Ret(nullptr);
		return Ret(-EBUSY);
	}
	return fn();
}

} // namespace bpftime

#endif // ROBUST_SHM_MUTEX_HPP
