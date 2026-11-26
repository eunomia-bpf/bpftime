// SPDX-License-Identifier: MIT
//
// Reservation lock semantics for the userspace ring buffer: multi-lap
// wraparound under concurrent in-process and forked producers, waiting on a
// live holder, and SIGKILL holder recovery via the robust process-shared
// mutex. Records drain through ringbuf::fetch_data(), which follows the
// same acquire-load / BUSY_BIT / roundup_len protocol as libbpf's
// ringbuf_process_ring().
#include "catch2/catch_test_macros.hpp"
#include "../common_def.hpp"
#include <bpf_map/userspace/ringbuf_map.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#if defined(__linux__)
#include <signal.h>
#include <sys/wait.h>
#endif

namespace
{
constexpr int MAP_FD = 7;
constexpr size_t RING_SIZE = 4096;
const size_t payload_sizes[] = { 16, 24, 48, 100, 200 };
struct record_header {
	uint64_t producer;
	uint64_t seq;
};

void produce(bpftime::ringbuf &rb, uint64_t producer, uint64_t seq)
{
	size_t size =
		payload_sizes[(producer + seq) % std::size(payload_sizes)];
	void *sample = nullptr;
	while ((sample = rb.reserve(size, MAP_FD)) == nullptr) {
		REQUIRE(errno == ENOSPC);
		sched_yield();
	}
	record_header hdr{ producer, seq };
	std::memcpy(sample, &hdr, sizeof(hdr));
	rb.submit(sample, false);
}

int collect(void *ctx, void *data, size_t)
{
	auto hdr = *static_cast<const record_header *>(data);
	static_cast<std::vector<std::vector<uint64_t>> *>(ctx)
		->at(hdr.producer)
		.push_back(hdr.seq);
	return 0;
}

template <typename Pred> void wait_until(Pred pred)
{
	auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(90);
	while (!pred()) {
		REQUIRE(std::chrono::steady_clock::now() < deadline);
		sched_yield();
	}
}
} // namespace

// Variable-size records cross many full ring laps while in-process threads
// (and, on Linux, forked processes sharing the segment) reserve
// concurrently; every record arrives exactly once with an intact sequence.
TEST_CASE("Ringbuf reserve survives multi-lap wraparound under contention",
	  "[ringbuf_map][reserve]")
{
	const std::string shm_name =
		"RingbufReserveWrapShm-" + std::to_string(getpid());
	shm_remove remover(shm_name.c_str());
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(), 1 << 20);
	const char *MAP_NAME = "ringbuf_reserve_mp_map";
	auto *map_ptr = shm.construct<bpftime::ringbuf_map_impl>(MAP_NAME)(
		RING_SIZE, shm);
	auto rb = map_ptr->create_impl_shared_ptr();
	constexpr size_t nr_threads = 4;
	constexpr uint64_t per_thread = 1200;
#if defined(__linux__)
	constexpr size_t nr_children = 4;
	constexpr uint64_t per_child = 400;
#else
	constexpr size_t nr_children = 0;
	constexpr uint64_t per_child = 0;
#endif
	constexpr size_t nr_producers = nr_threads + nr_children;
	const uint64_t expected =
		nr_threads * per_thread + nr_children * per_child;
#if defined(__linux__)
	auto child_run = +[](uint64_t producer, const char *segment,
			     const char *map_name) -> int {
		boost::interprocess::managed_shared_memory child_shm(
			boost::interprocess::open_only, segment);
		auto found =
			child_shm.find<bpftime::ringbuf_map_impl>(map_name);
		if (found.first == nullptr || found.second != 1)
			return 10;
		auto child_rb = found.first->create_impl_shared_ptr();
		for (uint64_t i = 0; i < per_child; ++i)
			produce(*child_rb, producer, i);
		return 0;
	};
	std::vector<pid_t> children;
	for (size_t c = 0; c < nr_children; ++c) {
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0)
			_exit(child_run(nr_threads + c, shm_name.c_str(),
					MAP_NAME));
		children.push_back(pid);
	}
#endif
	std::vector<std::thread> threads;
	for (size_t t = 0; t < nr_threads; ++t)
		threads.emplace_back([&, t] {
			for (uint64_t i = 0; i < per_thread; ++i)
				produce(*rb, t, i);
		});
	// Delivery of `expected` records proves every producer (threads and
	// children) finished, since nothing is delivered before production.
	std::vector<std::vector<uint64_t>> delivered(nr_producers);
	uint64_t total = 0;
	wait_until([&] {
		total += rb->fetch_data(collect, &delivered);
		return total >= expected;
	});
	for (auto &th : threads)
		th.join();
#if defined(__linux__)
	for (pid_t child : children) {
		int status = 0;
		REQUIRE(waitpid(child, &status, 0) == child);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
	}
#endif
	for (size_t p = 0; p < nr_producers; ++p) {
		auto &seqs = delivered[p];
		const uint64_t want = p >= nr_threads ? per_child : per_thread;
		REQUIRE(seqs.size() == want);
		std::sort(seqs.begin(), seqs.end());
		REQUIRE(std::adjacent_find(seqs.begin(), seqs.end()) ==
			seqs.end());
		REQUIRE(seqs.front() == 0);
		REQUIRE(seqs.back() == want - 1);
	}
}

// A descheduled live holder must not break other producers: reserve() blocks
// until release and then succeeds.
TEST_CASE("Ringbuf reserve waits for a live lock holder",
	  "[ringbuf_map][reserve]")
{
	const std::string shm_name =
		"RingbufReserveLiveHolderShm-" + std::to_string(getpid());
	shm_remove remover(shm_name.c_str());
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(), 1 << 20);
	bpftime::ringbuf_map_impl map(RING_SIZE, shm);
	auto rb = map.create_impl_shared_ptr();
	std::atomic<bool> holder_started{ false };
	std::thread holder([&] {
		rb->lock_for_testing();
		holder_started.store(true, std::memory_order_release);
		std::this_thread::sleep_for(std::chrono::milliseconds(25));
		rb->unlock_for_testing();
	});
	wait_until([&] { return holder_started.load(); });
	void *sample = rb->reserve(32, MAP_FD); // blocks until release
	holder.join();
	REQUIRE(sample != nullptr);
	record_header hdr{ 0, 1 };
	std::memcpy(sample, &hdr, sizeof(hdr));
	rb->submit(sample, false);
	std::vector<std::vector<uint64_t>> delivered(1);
	REQUIRE(rb->fetch_data(collect, &delivered) == 1);
	REQUIRE(delivered[0] == std::vector<uint64_t>{ 1 });
}

#if defined(__linux__)
// A producer SIGKILLed while holding the lock leaves an unreaped zombie.
// Kernel PTHREAD_MUTEX_ROBUST marks the mutex on exit; lock_and_recover()
// handles EOWNERDEAD via pthread_mutex_consistent() and reservation continues.
TEST_CASE("Ringbuf reserve recovers a SIGKILLed lock holder",
	  "[ringbuf_map][reserve]")
{
	const std::string shm_name =
		"RingbufReserveDeadHolderShm-" + std::to_string(getpid());
	shm_remove remover(shm_name.c_str());
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(), 1 << 20);
	bpftime::ringbuf_map_impl map(RING_SIZE, shm);
	auto rb = map.create_impl_shared_ptr();
	int pipefd[2];
	REQUIRE(pipe(pipefd) == 0);
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		close(pipefd[0]);
		rb->lock_for_testing();
		char ready = 'r';
		ssize_t written = write(pipefd[1], &ready, 1);
		(void)written;
		close(pipefd[1]);
		raise(SIGKILL);
		_exit(21);
	}
	close(pipefd[1]);
	char ready = 0;
	REQUIRE(read(pipefd[0], &ready, 1) == 1);
	close(pipefd[0]);
	int status = 0; // recover while the child is still an unreaped zombie
	wait_until([&] { return waitpid(child, &status, WNOHANG) == child; });
	REQUIRE(WIFSIGNALED(status));
	REQUIRE(WTERMSIG(status) == SIGKILL);
	for (uint64_t i = 1; i < 20; ++i)
		produce(*rb, 0, i);
	std::vector<std::vector<uint64_t>> delivered(1);
	uint64_t total = 0;
	wait_until([&] {
		total += rb->fetch_data(collect, &delivered);
		return total == 19;
	});
	auto &seqs = delivered[0];
	std::sort(seqs.begin(), seqs.end());
	REQUIRE(seqs.size() == 19);
	REQUIRE(seqs.front() == 1);
	REQUIRE(seqs.back() == 19);
}
#endif
