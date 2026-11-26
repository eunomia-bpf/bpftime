// SPDX-License-Identifier: MIT
#include "catch2/catch_test_macros.hpp"
#include "bpftime_handler.hpp"
#include "bpftime_shm.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <unistd.h>
#if defined(__linux__)
#include <signal.h>
#include <sys/wait.h>
#endif
using namespace boost::interprocess;
using namespace bpftime;
namespace
{
constexpr int MAP_FD = 1;
struct hash_map_fixture {
	const std::string shm_name;
	shm_remove remover;
	managed_shared_memory segment;
	const bpf_map_handler *map = nullptr;
	hash_map_fixture()
		: shm_name("MapLockRobustShm-" + std::to_string(getpid())),
		  remover(shm_name.c_str()),
		  segment(create_only, shm_name.c_str(), 8 << 20)
	{
		auto *mgr = segment.construct<handler_manager>(
			"map_lock_robust_manager")(segment, MIN_MAX_FD_COUNT);
		REQUIRE(mgr->set_handler(
				MAP_FD,
				bpf_map_handler(
					MAP_FD,
					(int)bpf_map_type::BPF_MAP_TYPE_HASH,
					sizeof(uint32_t), sizeof(uint64_t), 16,
					0, "robust_hash", segment),
				segment) == MAP_FD);
		auto *h = const_cast<bpf_map_handler *>(
			&std::get<bpf_map_handler>((*mgr)[MAP_FD]));
		REQUIRE(h->map_init(segment) == 0);
		map = h;
	}
};
} // namespace

TEST_CASE("Map helper hot path trylocks around a live holder while syscalls block",
	  "[map_lock][dead_owner]")
{
	hash_map_fixture fx;
	auto &map = *fx.map;
	std::atomic<bool> holder_holds{ false };
	std::atomic<bool> release_holder{ false };
	std::thread holder([&] {
		REQUIRE(map.get_map_lock().lock("holder"));
		holder_holds.store(true, std::memory_order_release);
		// Stay locked until the main thread has observed the helper
		// skips and explicitly releases us: no timing-based sleep, so a
		// descheduled waiter cannot miss the contention window.
		while (!release_holder.load(std::memory_order_acquire))
			std::this_thread::yield();
		map.get_map_lock().unlock();
	});
	while (!holder_holds.load(std::memory_order_acquire))
		std::this_thread::yield();

	// BPF helper entries (from_syscall == false) must skip immediately
	// with -EBUSY instead of blocking while a live holder holds the lock.
	// Host errno is preserved across the trylock fast path.
	std::atomic<int> skipped{ 0 }, errno_dirty{ 0 };
	std::array<std::thread, 2> waiters;
	for (int w = 0; w < 2; ++w)
		waiters[w] = std::thread([&, w] {
			uint32_t key = (uint32_t)(w + 1);
			uint64_t value = 0x1000 + (uint64_t)w;
			errno = 0;
			if (map.map_update_elem(&key, &value, 0, false) ==
			    -EBUSY) {
				skipped.fetch_add(1,
						  std::memory_order_relaxed);
				if (errno != 0)
					errno_dirty.fetch_add(1,
							     std::memory_order_relaxed);
			}
		});

	// Join all waiters before touching assertions, so a failed REQUIRE
	// never destroys joinable threads. The holder keeps the lock across
	// these joins, so every trylock deterministically sees contention.
	for (auto &t : waiters)
		t.join();
	REQUIRE(skipped.load() == 2);
	REQUIRE(errno_dirty.load() == 0);

	// The control-plane syscall entry keeps the blocking contract: it
	// waits for the holder to release and then succeeds.
	release_holder.store(true, std::memory_order_release);
	uint32_t sys_key = 9;
	uint64_t sys_value = 0x9000;
	REQUIRE(map.map_update_elem(&sys_key, &sys_value, 0, true) == 0);
	holder.join();
}

#if defined(__linux__)
TEST_CASE("Map lock fails open on a dead holder on both entry paths",
	  "[map_lock][dead_owner]")
{
	hash_map_fixture fx;
	auto &map = *fx.map;
	int pipefd[2]{ -1, -1 };
	REQUIRE(pipe(pipefd) == 0);
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		close(pipefd[0]);
		if (!map.get_map_lock().lock("child"))
			_exit(30);
		char ready = 'r';
		(void)write(pipefd[1], &ready, 1);
		close(pipefd[1]);
		raise(SIGKILL);
		_exit(31);
	}
	close(pipefd[1]);
	char ready = 0;
	REQUIRE(read(pipefd[0], &ready, 1) == 1);
	close(pipefd[0]);
	uint32_t key = 7;
	uint64_t value = 42;

	// Helper hot path: negative errno return, host errno left untouched.
	errno = 0;
	REQUIRE(map.map_update_elem(&key, &value, 0, false) == -EBUSY);
	REQUIRE(errno == 0);
	REQUIRE(map.map_lookup_elem(&key, false) == nullptr);

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGKILL);

	// Syscall control plane retains the blocking lock() convention:
	// -1 with errno = EBUSY on the permanently poisoned mutex.
	errno = 0;
	REQUIRE(map.map_update_elem(&key, &value, 0, true) == -1);
	REQUIRE(errno == EBUSY);
}
#endif
