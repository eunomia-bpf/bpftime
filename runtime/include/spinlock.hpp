#ifndef SPINLOCK_HPP
#define SPINLOCK_HPP

#include <atomic>
#include <cstddef>
#include <ctime>
#include <time.h>

class Spinlock {
    std::atomic<unsigned int> lock;
public:

    // Default constructor to initialize the atomic_flag
    Spinlock() : lock(0) {}

    // Deleted copy constructor to prevent copying
    Spinlock(const Spinlock&) = delete;
    Spinlock& operator=(const Spinlock&) = delete;

    Spinlock(Spinlock&& other) noexcept : lock() {
        if (other.lock.exchange(1, std::memory_order_acquire)) {
            lock.store(0, std::memory_order_release);
        }
    }

    Spinlock& operator=(Spinlock&& other) noexcept {
        if (this != &other) {
            if (other.lock.exchange(1, std::memory_order_acquire)) {
                lock.store(0, std::memory_order_release);
            }
        }
        return *this;
    }

    void spin_lock() {
        while(lock.load(std::memory_order_relaxed) || lock.exchange(1, std::memory_order_acquire)){
                __asm__ __volatile__ ("yield");
        }
    }

    void spin_unlock() {
        lock.store(0, std::memory_order_release);
    }
};

// Global functions to match pthread_spin_lock/unlock interface
inline void spin_lock(Spinlock* spinlock) {
    spinlock->spin_lock();
}

inline void spin_unlock(Spinlock* spinlock) {
    spinlock->spin_unlock();
}

#endif // SPINLOCK_HPP
