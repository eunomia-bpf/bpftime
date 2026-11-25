/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#ifndef ROBUST_SHM_MUTEX_HPP
#define ROBUST_SHM_MUTEX_HPP

#include "spdlog/spdlog.h"
#include <cerrno>
#include <pthread.h>
#include <stdexcept>
#include <string>

namespace bpftime
{

// Process-shared pthread mutex for SHM objects whose mutations can leave
// half-written state if a holder is killed (SIGKILL/OOM). On Linux it is
// ROBUST: a dead holder surfaces as EOWNERDEAD; we unlock without calling
// pthread_mutex_consistent(), permanently latching ENOTRECOVERABLE, and
// fail with errno=EBUSY rather than touching corrupt state.
class robust_shm_mutex {
	pthread_mutex_t mutex;

    public:
	robust_shm_mutex()
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

	bool lock(const char *label = nullptr)
	{
		int rc = pthread_mutex_lock(&mutex);
#ifdef __linux__
		if (rc == EOWNERDEAD) {
			// POSIX delivers EOWNERDEAD once before unlock() latches
			// ENOTRECOVERABLE; swallow logger exceptions so unlock()
			// always runs.
			try {
				SPDLOG_ERROR(
					"SHM robust mutex (guard '{}') owner died (EOWNERDEAD); latching ENOTRECOVERABLE, future lockers fail with EBUSY",
					label ? label : "unlabeled");
			} catch (...) {
			}
			pthread_mutex_unlock(&mutex);
			errno = EBUSY;
			return false;
		}
		if (rc == ENOTRECOVERABLE) {
			errno = EBUSY;
			return false;
		}
#endif
		if (rc != 0) {
			errno = rc;
			return false;
		}
		return true;
	}

	void unlock() { pthread_mutex_unlock(&mutex); }
};

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

} // namespace bpftime

#endif // ROBUST_SHM_MUTEX_HPP
