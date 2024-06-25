#ifndef SPINLOCK_WRAPPER_HPP
#define SPINLOCK_WRAPPER_HPP

#include <new>
#include <pthread.h>
#include <atomic>

// Include the custom spinlock implementation
#include "spinlock.hpp"

// Define the custom spinlock type and functions only if the standard library is not available
#if !defined(_POSIX_SPIN_LOCKS) || _POSIX_SPIN_LOCKS <= 0
typedef Spinlock pthread_spinlock_t; // Define custom spinlock type

// Non-volatile version
inline int pthread_spin_init(pthread_spinlock_t* lock, [[maybe_unused]]int pshared) {
    new(lock) pthread_spinlock_t(); // Placement new to initialize the Spinlock object
    return 0;
}

inline int pthread_spin_destroy(pthread_spinlock_t* lock) {
    lock->~Spinlock(); // Explicitly call the destructor
    return 0;
}

inline int pthread_spin_lock(pthread_spinlock_t* lock) {
    spin_lock(lock);
    return 0;
}

inline int pthread_spin_unlock(pthread_spinlock_t* lock) {
    spin_unlock(lock);
    return 0;
}

// Volatile version
inline int pthread_spin_init(volatile pthread_spinlock_t* lock, int pshared) {
    return pthread_spin_init(const_cast<pthread_spinlock_t*>(lock), pshared);
}

inline int pthread_spin_destroy(volatile pthread_spinlock_t* lock) {
    return pthread_spin_destroy(const_cast<pthread_spinlock_t*>(lock));
}

inline int pthread_spin_lock(volatile pthread_spinlock_t* lock) {
    return pthread_spin_lock(const_cast<pthread_spinlock_t*>(lock));
}

inline int pthread_spin_unlock(volatile pthread_spinlock_t* lock) {
    return pthread_spin_unlock(const_cast<pthread_spinlock_t*>(lock));
}

#else
// Ensure the volatile versions of the standard library functions

inline int pthread_spin_init(volatile pthread_spinlock_t* lock, int pshared) {
    return pthread_spin_init(const_cast<pthread_spinlock_t*>(lock), pshared);
}

inline int pthread_spin_destroy(volatile pthread_spinlock_t* lock) {
    return pthread_spin_destroy(const_cast<pthread_spinlock_t*>(lock));
}

inline int pthread_spin_lock(volatile pthread_spinlock_t* lock) {
    return pthread_spin_lock(const_cast<pthread_spinlock_t*>(lock));
}

inline int pthread_spin_unlock(volatile pthread_spinlock_t* lock) {
    return pthread_spin_unlock(const_cast<pthread_spinlock_t*>(lock));
}

#endif // _POSIX_SPIN_LOCKS

#endif // SPINLOCK_WRAPPER_HPP
