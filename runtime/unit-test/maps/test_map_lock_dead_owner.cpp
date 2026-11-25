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

TEST_CASE("Map lock serializes contention and fails open on dead holder",
	  "[map_lock][dead_owner]")
{
	hash_map_fixture fx;
	auto &map = *fx.map;
	std::atomic<bool> started{ false };
	std::thread holder([&] {
		REQUIRE(map.get_map_lock().lock("holder"));
		started.store(true, std::memory_order_release);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		map.get_map_lock().unlock();
	});
	while (!started.load(std::memory_order_acquire))
		std::this_thread::yield();
	std::atomic<int> ok{ 0 }, busy{ 0 };
	std::array<std::thread, 2> waiters;
	for (int w = 0; w < 2; ++w)
		waiters[w] = std::thread([&, w] {
			uint32_t key = (uint32_t)(w + 1);
			uint64_t value = 0x1000 + (uint64_t)w;
			errno = 0;
			if (map.map_update_elem(&key, &value, 0, false) == 0)
				ok.fetch_add(1, std::memory_order_relaxed);
			else if (errno == EBUSY)
				busy.fetch_add(1, std::memory_order_relaxed);
		});
	for (auto &t : waiters)
		t.join();
	holder.join();
	REQUIRE(busy.load() == 0);
	REQUIRE(ok.load() == 2);

#if defined(__linux__)
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
	errno = 0;
	REQUIRE(map.map_update_elem(&key, &value, 0, false) == -1);
	REQUIRE(errno == EBUSY);
	errno = 0;
	REQUIRE(map.map_lookup_elem(&key, false) == nullptr);
	REQUIRE(errno == EBUSY);
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGKILL);
	errno = 0;
	REQUIRE(map.map_update_elem(&key, &value, 0, false) == -1);
	REQUIRE(errno == EBUSY);
#endif
}
