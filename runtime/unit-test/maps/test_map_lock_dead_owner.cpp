// SPDX-License-Identifier: MIT
#include "catch2/catch_test_macros.hpp"
#include "bpftime_handler.hpp"
#include "bpftime_shm.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#if defined(__linux__)
#include <signal.h>
#include <sys/wait.h>
#endif
using namespace boost::interprocess;
using namespace bpftime;
namespace
{
constexpr int MAP_FD = 1;
struct lru_map_fixture {
	const std::string shm_name;
	shm_remove remover;
	managed_shared_memory segment;
	const bpf_map_handler *map = nullptr;
	lru_map_fixture()
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
					(int)bpf_map_type::BPF_MAP_TYPE_LRU_HASH,
					sizeof(uint32_t), sizeof(uint64_t), 16,
					0, "robust_lru", segment),
				segment) == MAP_FD);
		auto *h = const_cast<bpf_map_handler *>(
			&std::get<bpf_map_handler>((*mgr)[MAP_FD]));
		REQUIRE(h->map_init(segment) == 0);
		map = h;
	}
};
} // namespace

TEST_CASE("Container map helper trylocks while HASH/PERCPU_HASH/PERF/BLOOM avoid global map_lock",
	  "[map_lock][dead_owner]")
{
	lru_map_fixture fx;
	auto &map = *fx.map;
	std::atomic<bool> started{ false }, release{ false };
	std::thread holder([&] {
		REQUIRE(map.get_map_lock().lock("holder"));
		started.store(true, std::memory_order_release);
		while (!release.load(std::memory_order_acquire))
			std::this_thread::yield();
		map.get_map_lock().unlock();
	});
	while (!started.load(std::memory_order_acquire))
		std::this_thread::yield();
	const uint64_t skips_before = bpftime::get_shm_trylock_skip_count();
	std::atomic<int> skipped{ 0 }, errno_dirty{ 0 };
	std::array<std::thread, 2> waiters;
	for (int w = 0; w < 2; ++w)
		waiters[w] = std::thread([&, w] {
			uint32_t key = (uint32_t)(w + 1);
			uint64_t value = 0x1000 + (uint64_t)w;
			errno = 0;
			if (map.map_update_elem(&key, &value, 0, false) ==
			    -EBUSY) {
				skipped.fetch_add(1, std::memory_order_relaxed);
				errno_dirty.fetch_add(errno != 0,
						      std::memory_order_relaxed);
			}
		});
	for (auto &t : waiters)
		t.join();
	const uint64_t skips_after = bpftime::get_shm_trylock_skip_count();
	release.store(true, std::memory_order_release);
	holder.join();
	uint32_t sys_key = 9;
	uint64_t sys_val = 0x9000;
	REQUIRE(skipped.load() == 2);
	REQUIRE(errno_dirty.load() == 0);
	REQUIRE(skips_after - skips_before == 2);
	REQUIRE(map.map_update_elem(&sys_key, &sys_val, 0, true) == 0);

	bpf_map_handler hmap(2, (int)bpf_map_type::BPF_MAP_TYPE_HASH, 4, 8, 64,
			     0, "hmap", fx.segment);
	bpf_map_handler pcpu(3, (int)bpf_map_type::BPF_MAP_TYPE_PERCPU_HASH, 4,
			     8, 64, 0, "pcpu", fx.segment);
	bpf_map_handler perf(4, (int)bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY,
			     4, 4, 8, 0, "perf", fx.segment);
	bpf_map_attr ba{};
	ba.type = (int)bpf_map_type::BPF_MAP_TYPE_BLOOM_FILTER;
	ba.value_size = 8;
	ba.max_ents = 32;
	ba.map_extra = 3;
	bpf_map_handler bloom(5, "bloom", fx.segment, ba);
	for (auto *m : { &hmap, &pcpu, &perf, &bloom }) {
		REQUIRE(m->map_init(fx.segment) == 0);
		REQUIRE(m->get_map_lock().lock("global-held"));
	}
	REQUIRE(hmap.map_update_elem(&sys_key, &sys_val, 0, false) == 0);
	REQUIRE(pcpu.map_update_elem(&sys_key, &sys_val, 0, false) == 0);
	int32_t slot0 = 0, pfd = 42;
	REQUIRE(perf.map_update_elem(&slot0, &pfd, 0, true) == 0);
	REQUIRE(*(const int32_t *)perf.map_lookup_elem(&slot0, false) == 42);
	REQUIRE(bloom.map_push_elem(&sys_val, 0, false) == 0);
	REQUIRE(bloom.map_peek_elem(&sys_val, false) == 0);
	for (auto *m : { &hmap, &pcpu, &perf, &bloom })
		m->get_map_lock().unlock();

	std::atomic<int> false_nulls{ 0 };
	std::vector<std::thread> readers;
	for (int t = 0; t < 8; ++t)
		readers.emplace_back([&, t] {
			for (int i = 0; i < 2000; ++i) {
				uint32_t k = (uint32_t)(100 + t);
				uint64_t v = (uint64_t)i;
				(void)hmap.map_update_elem(&k, &v, 0, false);
				if (!hmap.map_lookup_elem(&sys_key, false) ||
				    !pcpu.map_lookup_elem(&sys_key, false))
					false_nulls.fetch_add(1);
			}
		});
	for (auto &th : readers)
		th.join();
	REQUIRE(false_nulls.load() == 0);
	for (auto *m : { &hmap, &pcpu, &perf, &bloom })
		m->map_free(fx.segment);

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
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGKILL);
	uint32_t key = 7;
	uint64_t value = 42;
	errno = 0;
	REQUIRE(map.map_update_elem(&key, &value, 0, false) == -EBUSY);
	REQUIRE(errno == 0);
	REQUIRE(map.map_lookup_elem(&key, false) == nullptr);
	REQUIRE(errno == 0);
	REQUIRE(map.map_update_elem(&key, &value, 0, true) == -1);
	REQUIRE(errno == EBUSY);
#endif
}
