#ifndef SPINLOCK_HPP
#define SPINLOCK_HPP

#include <atomic>

class Spinlock {
public:
    std::atomic_flag lock = ATOMIC_FLAG_INIT;

    // Default constructor to initialize the atomic_flag
    Spinlock() = default; 

    // // Deleted copy constructor to prevent copying
    Spinlock(const Spinlock&) = delete;
    Spinlock& operator=(const Spinlock&) = delete;

    Spinlock(Spinlock&& other) noexcept : lock() {
        if (other.lock.test_and_set(std::memory_order_acquire)) {
            lock.clear(std::memory_order_release);
        }
    }

    Spinlock& operator=(Spinlock&& other) noexcept {
        if (this != &other) {
            if (other.lock.test_and_set(std::memory_order_acquire)) {
                lock.clear(std::memory_order_release);
            }
        }
        return *this;
    }

    void spin_lock() {
        while (lock.test_and_set(std::memory_order_acquire)) {
            // busy-wait
        }
    }

    void spin_unlock() {
        lock.clear(std::memory_order_release);
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
