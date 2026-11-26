/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include <catch2/catch_test_macros.hpp>

#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "ringbuf_eventfd.hpp"
#include "syscall_context.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <linux/bpf.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#ifndef BPF_RB_NO_WAKEUP
#define BPF_RB_NO_WAKEUP (1ULL << 0)
#endif
#ifndef BPF_RB_FORCE_WAKEUP
#define BPF_RB_FORCE_WAKEUP (1ULL << 1)
#endif
namespace
{
using clock_type = std::chrono::steady_clock;

struct eventfd_test_fixture {
	syscall_context ctx;
	const std::string shm_name;
	int rb_fd = -1;
	int epfd = -1;
	static constexpr uint64_t epoll_data_marker = 0x7177ULL;

	eventfd_test_fixture()
		: shm_name("bpftime-rb-eventfd-test-" +
			   std::to_string(getpid()))
	{
		setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
		ctx.try_startup();
		union bpf_attr attr{};
		attr.map_type = BPF_MAP_TYPE_RINGBUF;
		attr.max_entries = 64U * getpagesize();
		rb_fd = ctx.handle_sysbpf(BPF_MAP_CREATE, &attr, sizeof(attr));
		REQUIRE(rb_fd >= 0);
		epfd = ctx.handle_epoll_create1(0);
		REQUIRE(epfd >= 0);
		epoll_event evt{ .events = EPOLLIN,
				 .data = { .u64 = epoll_data_marker } };
		REQUIRE(ctx.handle_epoll_ctl(epfd, EPOLL_CTL_ADD, rb_fd,
					     &evt) == 0);
	}

	~eventfd_test_fixture()
	{
		if (epfd >= 0)
			ctx.handle_close(epfd);
		if (rb_fd >= 0)
			ctx.handle_close(rb_fd);
		bpftime_destroy_global_shm();
		bpftime_remove_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_NO_CREATE);
		unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	}

	void submit(uint64_t value, uint64_t flags = 0)
	{
		auto *buf = bpftime_ringbuf_reserve(rb_fd, sizeof(value));
		REQUIRE(buf != nullptr);
		int saved = bpftime_protect_open_write_window();
		*static_cast<uint64_t *>(buf) = value;
		bpftime_protect_restore(saved);
		bpftime_ringbuf_submit_with_flags(rb_fd, buf, false, flags);
	}

	void drain_all()
	{
		// bpftime_poll_from_ringbuf returns the number of records
		// fetched (or a negative errno), so loop until the ring is
		// empty rather than expecting zero from the first call.
		int drained;
		do {
			drained = bpftime_poll_from_ringbuf(
				rb_fd, nullptr,
				[](void *, void *, size_t) { return 0; });
			REQUIRE(drained >= 0);
		} while (drained > 0);
	}
};

int stop_after_one(void *ctx, void *, size_t)
{
	auto *count = static_cast<int *>(ctx);
	++*count;
	// A negative callback result stops fetch_data after committing the
	// current record, simulating a consumer that partially drains.
	return *count == 1 ? -1 : 0;
}
} // namespace

TEST_CASE("ringbuf eventfd: cross-process submit wakes blocked epoll_wait",
	  "[epoll][ringbuf-eventfd]")
{
	eventfd_test_fixture f;
	auto *submit_ns = static_cast<std::atomic<int64_t> *>(
		mmap(nullptr, sizeof(int64_t), PROT_READ | PROT_WRITE,
		     MAP_SHARED | MAP_ANONYMOUS, -1, 0));
	REQUIRE(submit_ns != MAP_FAILED);

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		// The forked child inherited the creator's cache entry; drop
		// it so the wakeup fd is re-resolved through the abstract
		// AF_UNIX SCM_RIGHTS exporter, exactly as an injected agent
		// process must do (no fd inheritance, no /proc fd access).
		bpftime_ringbuf_drop_local_eventfd(f.rb_fd);
		usleep(20 * 1000);
		auto *buf = bpftime_ringbuf_reserve(f.rb_fd, sizeof(uint64_t));
		if (buf == nullptr)
			_exit(10);
		int saved = bpftime_protect_open_write_window();
		*static_cast<uint64_t *>(buf) = 7;
		bpftime_protect_restore(saved);
		submit_ns->store(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				clock_type::now().time_since_epoch())
				.count());
		bpftime_ringbuf_submit_with_flags(f.rb_fd, buf, false, 0);
		_exit(0);
	}

	std::array<epoll_event, 4> evts{};
	const int rc =
		f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(), 5000);
	const auto wake_ns =
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			clock_type::now().time_since_epoch())
			.count();
	REQUIRE(rc == 1);
	REQUIRE(evts[0].data.u64 == eventfd_test_fixture::epoll_data_marker);
	const int64_t latency_us = (wake_ns - submit_ns->load()) / 1000;
	INFO("cross-process eventfd wakeup latency: " << latency_us << " us");
	REQUIRE(latency_us >= 0);
	REQUIRE(latency_us < 5000);

	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
	munmap(submit_ns, sizeof(int64_t));
	f.drain_all();
}

TEST_CASE("ringbuf eventfd: level-triggered until fully drained",
	  "[epoll][ringbuf-eventfd]")
{
	eventfd_test_fixture f;
	f.submit(1);
	f.submit(2);

	std::array<epoll_event, 4> evts{};
	REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(),
					100) == 1);

	// Partial drain: one record consumed, one remains.
	int consumed = 0;
	REQUIRE(bpftime_poll_from_ringbuf(f.rb_fd, &consumed, stop_after_one) <
		0);
	REQUIRE(consumed == 1);

	// The ringbuf still holds data, so a non-blocking poll stays ready.
	evts.fill(epoll_event{});
	REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(), 0) ==
		1);
	REQUIRE(evts[0].data.u64 == eventfd_test_fixture::epoll_data_marker);

	// Full drain: the next blocking wait must consume the stale eventfd
	// counter and block for the whole timeout instead of waking on it.
	f.drain_all();
	const auto t0 = clock_type::now();
	evts.fill(epoll_event{});
	REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(),
					200) == 0);
	const auto elapsed_ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(
			clock_type::now() - t0)
			.count();
	INFO("post-drain wait returned after " << elapsed_ms << " ms");
	REQUIRE(elapsed_ms >= 100);
}

TEST_CASE("ringbuf eventfd: NO_WAKEUP suppresses, FORCE_WAKEUP signals",
	  "[epoll][ringbuf-eventfd]")
{
	eventfd_test_fixture f;
	std::array<epoll_event, 4> evts{};

	int rc = 42;
	std::thread suppressed([&] {
		rc = f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(),
					     200);
	});
	usleep(30 * 1000);
	f.submit(1, BPF_RB_NO_WAKEUP);
	suppressed.join();
	REQUIRE(rc == 0);
	f.drain_all();

	rc = 42;
	std::thread forced([&] {
		rc = f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(),
					     2000);
	});
	usleep(30 * 1000);
	// The consumer is caught up, so the adaptive default would also wake;
	// FORCE_WAKEUP guarantees the eventfd signal regardless of positions.
	const auto t0 = clock_type::now();
	f.submit(2, BPF_RB_FORCE_WAKEUP);
	forced.join();
	const auto wake_ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(
			clock_type::now() - t0)
			.count();
	REQUIRE(rc == 1);
	REQUIRE(wake_ms < 500);
	f.drain_all();
}

TEST_CASE("ringbuf eventfd: timeout=0 polls once and SIGINT gives EINTR",
	  "[epoll][ringbuf-eventfd]")
{
	eventfd_test_fixture f;
	std::array<epoll_event, 4> evts{};

	// Empty ringbuf, non-blocking: returns immediately with no event.
	const auto t0 = clock_type::now();
	REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(), 0) ==
		0);
	REQUIRE(std::chrono::duration_cast<std::chrono::milliseconds>(
			clock_type::now() - t0)
			.count() < 100);

	struct sigaction sa{};
	sa.sa_handler = [](int) {};
	sigemptyset(&sa.sa_mask);
	struct sigaction old_sa{};
	REQUIRE(sigaction(SIGINT, &sa, &old_sa) == 0);

	int rc = 42;
	int saved_errno = 0;
	std::thread waiter([&] {
		errno = 0;
		rc = f.ctx.handle_epoll_wait(f.epfd, evts.data(), evts.size(),
					     -1);
		saved_errno = errno;
	});
	usleep(30 * 1000);
	pthread_kill(waiter.native_handle(), SIGINT);
	waiter.join();
	REQUIRE(rc == -1);
	REQUIRE(saved_errno == EINTR);
	sigaction(SIGINT, &old_sa, nullptr);
}

TEST_CASE("ringbuf eventfd: wedged exporter socket falls back on timeout",
	  "[epoll][ringbuf-eventfd]")
{
	// Bind before try_startup so our non-accepting listener owns the
	// abstract exporter address, standing in for a stopped exporter.
	const std::string shm = "bpftime-rb-wedge-" + std::to_string(getpid());
	setenv("BPFTIME_GLOBAL_SHM_NAME", shm.c_str(), 1);
	bpftime_destroy_global_shm();
	struct sockaddr_un addr { .sun_family = AF_UNIX };
	constexpr char prefix[] = "\0bpftime-mapfd-";
	const size_t plen = sizeof(prefix) - 1;
	std::memcpy(addr.sun_path, prefix, plen);
	std::memcpy(addr.sun_path + plen, shm.data(), shm.size());
	socklen_t alen = static_cast<socklen_t>(
		offsetof(struct sockaddr_un, sun_path) + plen + shm.size());
	int wedge = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	REQUIRE(bind(wedge, reinterpret_cast<struct sockaddr *>(&addr),
		     alen) == 0);
	REQUIRE(listen(wedge, 1) == 0);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	int status = 0;
	clock_type::duration elapsed{};
	{
		syscall_context ctx;
		ctx.try_startup();
		union bpf_attr attr{ .map_type = BPF_MAP_TYPE_RINGBUF };
		attr.max_entries = 64U * getpagesize();
		int rb_fd = ctx.handle_sysbpf(BPF_MAP_CREATE, &attr, sizeof(attr));
		REQUIRE(rb_fd >= 0);
		int epfd = ctx.handle_epoll_create1(0);
		REQUIRE(epfd >= 0);
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0) {
			bpftime_ringbuf_drop_local_eventfd(rb_fd);
			epoll_event evt{ .events = EPOLLIN };
			if (ctx.handle_epoll_ctl(epfd, EPOLL_CTL_ADD, rb_fd, &evt) != 0)
				_exit(12);
			uint32_t sample = 0x5a5a5a5aU;
			void *slot = bpftime_ringbuf_reserve(rb_fd, sizeof(sample));
			if (!slot)
				_exit(13);
			int saved = bpftime_protect_open_write_window();
			std::memcpy(slot, &sample, sizeof(sample));
			bpftime_protect_restore(saved);
			bpftime_ringbuf_submit(rb_fd, slot, 0);
			epoll_event out{};
			_exit(ctx.handle_epoll_wait(epfd, &out, 1, 50) == 1 ? 0 : 14);
		}
		const auto t0 = clock_type::now();
		REQUIRE(waitpid(pid, &status, 0) == pid);
		elapsed = clock_type::now() - t0;
		ctx.handle_close(epfd);
		ctx.handle_close(rb_fd);
	}
	const auto ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
			.count();
	close(wedge);
	bpftime_destroy_global_shm();
	bpftime_remove_global_shm();
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_NO_CREATE);
	unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	REQUIRE((WIFEXITED(status) && WEXITSTATUS(status) == 0));
	REQUIRE(ms < 1500);
}

extern "C" uint64_t bpf_ringbuf_submit(uint64_t data, uint64_t flags, uint64_t,
				       uint64_t, uint64_t);
extern "C" uint64_t bpf_ringbuf_discard(uint64_t data, uint64_t flags, uint64_t,
					uint64_t, uint64_t);

// Dekker lost-wakeup regression: a producer that commits (or discards a head
// record blocking already-committed records) while the consumer is finishing
// the previous record and parking in epoll_wait must still deliver an eventfd
// wakeup.
TEST_CASE("ringbuf eventfd: commit before wakeup keeps parked consumer live",
	  "[epoll][ringbuf-eventfd]")
{
	SECTION("unknown helper flags report EINVAL without stranding BUSY bit")
	{
		eventfd_test_fixture f;
		auto *r1 = bpftime_ringbuf_reserve(f.rb_fd, sizeof(uint64_t));
		auto *r2 = bpftime_ringbuf_reserve(f.rb_fd, sizeof(uint64_t));
		REQUIRE(r1 != nullptr);
		REQUIRE(r2 != nullptr);
		int saved = bpftime_protect_open_write_window();
		*static_cast<uint64_t *>(r1) = 0x1111ULL;
		*static_cast<uint64_t *>(r2) = 0x2222ULL;
		bpftime_protect_restore(saved);

		REQUIRE(bpf_ringbuf_discard(reinterpret_cast<uint64_t>(r1),
					    0x100ULL, 0, 0,
					    0) == static_cast<uint64_t>(-EINVAL));
		REQUIRE(bpf_ringbuf_submit(reinterpret_cast<uint64_t>(r2),
					   0x200ULL, 0, 0,
					   0) == static_cast<uint64_t>(-EINVAL));

		uint64_t seen = 0;
		int fetched = bpftime_poll_from_ringbuf(
			f.rb_fd, &seen, [](void *ctx, void *data, size_t len) {
				if (len == sizeof(uint64_t))
					*static_cast<uint64_t *>(ctx) =
						*static_cast<uint64_t *>(data);
				return 0;
			});
		REQUIRE(fetched == 1);
		REQUIRE(seen == 0x2222ULL);
	}

	SECTION("discarding head BUSY record wakes consumer for queued record")
	{
		eventfd_test_fixture f;
		std::array<epoll_event, 4> evts{};
		auto *r1 = bpftime_ringbuf_reserve(f.rb_fd, sizeof(uint64_t));
		auto *r2 = bpftime_ringbuf_reserve(f.rb_fd, sizeof(uint64_t));
		REQUIRE(r1 != nullptr);
		REQUIRE(r2 != nullptr);
		{
			int saved = bpftime_protect_open_write_window();
			*static_cast<uint64_t *>(r2) = 42;
			bpftime_protect_restore(saved);
		}
		// r2 commits behind BUSY r1: adaptive wakeup does not fire yet.
		bpftime_ringbuf_submit_with_flags(f.rb_fd, r2, false, 0);
		REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(),
						evts.size(), 0) == 0);
		// Discarding r1 at consumer_pos must wake the consumer for r2.
		bpftime_ringbuf_submit_with_flags(f.rb_fd, r1, true, 0);
		REQUIRE(f.ctx.handle_epoll_wait(f.epfd, evts.data(),
						evts.size(), 0) == 1);
		int seen = 0;
		REQUIRE(bpftime_poll_from_ringbuf(
				f.rb_fd, &seen,
				[](void *ctx, void *, size_t) {
					++(*static_cast<int *>(ctx));
					return 0;
				}) == 1);
		REQUIRE(seen == 1);
	}

	SECTION("concurrent producer/consumer bursts never lose epoll wakeup")
	{
		constexpr int kBursts = 2000;
		constexpr int kMaxInFlight = 64;
		constexpr int kWaitTimeoutMs = 2000;
		eventfd_test_fixture f;
		std::array<epoll_event, 4> evts{};
		std::atomic<bool> consumer_ready{ false };
		std::atomic<int> consumed{ 0 };
		std::atomic<bool> consumer_failed{ false };
		std::atomic<bool> producer_failed{ false };

		std::thread consumer([&] {
			consumer_ready.store(true, std::memory_order_release);
			int total = 0;
			while (total < kBursts &&
			       !producer_failed.load(
				       std::memory_order_acquire)) {
				evts.fill(epoll_event{});
				int rc = f.ctx.handle_epoll_wait(
					f.epfd, evts.data(), evts.size(),
					kWaitTimeoutMs);
				if (rc != 1 ||
				    evts[0].data.u64 !=
					    eventfd_test_fixture::
						    epoll_data_marker) {
					consumer_failed.store(
						true,
						std::memory_order_release);
					return;
				}
				int drained;
				do {
					drained = bpftime_poll_from_ringbuf(
						f.rb_fd, nullptr,
						[](void *, void *, size_t) {
							return 0;
						});
					if (drained < 0) {
						consumer_failed.store(
							true,
							std::memory_order_release);
						return;
					}
					total += drained;
				} while (drained > 0);
				consumed.store(total,
					       std::memory_order_release);
			}
		});

		while (!consumer_ready.load(std::memory_order_acquire))
			std::this_thread::yield();

		const auto deadline =
			clock_type::now() + std::chrono::seconds(15);
		for (int i = 0; i < kBursts; ++i) {
			while (i - consumed.load(std::memory_order_acquire) >=
			       kMaxInFlight) {
				if (consumer_failed.load(
					    std::memory_order_acquire) ||
				    clock_type::now() > deadline) {
					producer_failed.store(
						true,
						std::memory_order_release);
					break;
				}
				std::this_thread::yield();
			}
			if (producer_failed.load(std::memory_order_acquire) ||
			    consumer_failed.load(std::memory_order_acquire))
				break;
			uint64_t value = static_cast<uint64_t>(i) + 1;
			auto *buf =
				bpftime_ringbuf_reserve(f.rb_fd, sizeof(value));
			if (buf == nullptr) {
				producer_failed.store(
					true, std::memory_order_release);
				break;
			}
			int saved = bpftime_protect_open_write_window();
			*static_cast<uint64_t *>(buf) = value;
			bpftime_protect_restore(saved);
			bpftime_ringbuf_submit_with_flags(f.rb_fd, buf, false,
							  0);
			std::this_thread::yield();
		}

		consumer.join();
		REQUIRE_FALSE(producer_failed.load(std::memory_order_acquire));
		REQUIRE_FALSE(consumer_failed.load(std::memory_order_acquire));
		REQUIRE(consumed.load(std::memory_order_acquire) == kBursts);
		f.drain_all();
	}
}
#endif
