#ifndef _BPFTIME_LOCK_GUARD_HPP
#define _BPFTIME_LOCK_GUARD_HPP

#include <pthread.h>
namespace bpftime
{
namespace shm_common
{
// lock guad for RAII
class bpftime_lock_guard {
    private:
	volatile pthread_spinlock_t &spinlock;

    public:
	explicit bpftime_lock_guard(volatile pthread_spinlock_t &spinlock)
		: spinlock(spinlock)
	{
		pthread_spin_lock(&spinlock);
	}
	~bpftime_lock_guard()
	{
		pthread_spin_unlock(&spinlock);
	}

	// Delete copy constructor and assignment operator
	bpftime_lock_guard(const bpftime_lock_guard &) = delete;
	bpftime_lock_guard &operator=(const bpftime_lock_guard &) = delete;
};

} // namespace shm_common
} // namespace bpftime
#endif
