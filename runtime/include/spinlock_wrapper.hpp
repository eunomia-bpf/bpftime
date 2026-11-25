#ifndef SPINLOCK_WRAPPER_HPP
#define SPINLOCK_WRAPPER_HPP

#include <new>
#include <pthread.h>
#include <atomic>
#include <type_traits>

// Include the custom spinlock implementation
#include "spinlock.hpp"

// Define the custom spinlock type and functions only if the standard library
// does not provide spinlocks
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
// The standard library provides the type and the functions, and on every
// libc where pthread_spinlock_t is already volatile-qualified (glibc's
// `volatile int`) nothing is missing. Forwarding from the volatile form to
// the plain one is only a distinct overload when the element type is not
// volatile; written unconditionally, the const_cast produces the same
// parameter type and the wrapper calls itself until the stack overflows.
template <typename T>
inline constexpr bool spinlock_element_is_volatile =
	std::is_volatile<typename std::remove_pointer<T>::type>::value;

template <typename T,
	  typename = std::enable_if_t<!spinlock_element_is_volatile<T>>>
inline int pthread_spin_init(volatile T *lock, int pshared)
{
	return pthread_spin_init(const_cast<T *>(lock), pshared);
}

template <typename T,
	  typename = std::enable_if_t<!spinlock_element_is_volatile<T>>>
inline int pthread_spin_destroy(volatile T *lock)
{
	return pthread_spin_destroy(const_cast<T *>(lock));
}

template <typename T,
	  typename = std::enable_if_t<!spinlock_element_is_volatile<T>>>
inline int pthread_spin_lock(volatile T *lock)
{
	return pthread_spin_lock(const_cast<T *>(lock));
}

template <typename T,
	  typename = std::enable_if_t<!spinlock_element_is_volatile<T>>>
inline int pthread_spin_unlock(volatile T *lock)
{
	return pthread_spin_unlock(const_cast<T *>(lock));
}

#endif // _POSIX_SPIN_LOCKS

#endif // SPINLOCK_WRAPPER_HPP
