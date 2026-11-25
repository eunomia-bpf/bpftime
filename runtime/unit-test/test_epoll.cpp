/* SPDX-License-Identifier: MIT */
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#if __linux__
#include "bpftime_shm.hpp"
#include "common_def.hpp"
#include <boost/scope_exit.hpp>
#include <array>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <linux/bpf.h>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <vector>

TEST_CASE("Epoll returns each ready registration once and restores signals",
	  "[epoll]")
{
	const bool ringbuf = GENERATE(false, true);
	const bool mixed_mask = GENERATE(false, true);
	struct wait_case {
		const char *name;
		int sources;
		int ready;
		int capacity;
		int timeout;
		int expected;
	};
	const auto test = GENERATE(
		wait_case{ "empty instance", 0, 0, 4, 0, 0 },
		wait_case{ "empty source", 1, 0, 4, 0, 0 },
		wait_case{ "ready poll", 1, 1, 4, 0, 1 },
		wait_case{ "ready finite wait", 1, 1, 4, 1000, 1 },
		wait_case{ "two ready polls", 2, 2, 4, 0, 2 },
		wait_case{ "two ready finite waits", 2, 2, 4, 1000, 2 },
		wait_case{ "poll capacity", 2, 2, 1, 0, 1 },
		wait_case{ "finite wait capacity", 2, 2, 1, 1000, 1 },
		wait_case{ "ready indefinite wait", 1, 1, 4, -1, 1 },
		wait_case{ "empty finite wait", 1, 0, 4, 1, 0 });
	CAPTURE(ringbuf, mixed_mask, test.name);

	sigset_t original;
	REQUIRE(pthread_sigmask(SIG_SETMASK, nullptr, &original) == 0);
	sigset_t before = original;
	REQUIRE(sigdelset(&before, SIGINT) == 0);
	REQUIRE(sigdelset(&before, SIGTERM) == 0);
	if (mixed_mask) {
		REQUIRE(sigaddset(&before, SIGTERM) == 0);
		REQUIRE(sigaddset(&before, SIGUSR1) == 0);
	}
	const char *old_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	const bool had_name = old_name != nullptr;
	const std::string previous_name = old_name ? old_name : "";
	const std::string name = "EpollTestShm-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	std::vector<int> fds;
	bool initialized = false;
	BOOST_SCOPE_EXIT_ALL(&)
	{
		pthread_sigmask(SIG_SETMASK, &original, nullptr);
		if (initialized) {
			for (int fd : fds) {
				if (fd >= 0) {
					bpftime_close(fd);
					close(fd);
				}
			}
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
		if (had_name)
			setenv("BPFTIME_GLOBAL_SHM_NAME", previous_name.c_str(),
			       1);
		else
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	};
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	initialized = true;
	const int epoll_fd = bpftime_epoll_create();
	fds.push_back(epoll_fd);
	REQUIRE(epoll_fd >= 0);
	for (int i = 0; i < test.sources; ++i) {
		const int fd =
			ringbuf ? bpftime_maps_create(
					  -1, "epoll_ring",
					  { .type = BPF_MAP_TYPE_RINGBUF,
					    .max_ents = static_cast<uint32_t>(
						    getpagesize()) }) :
				  bpftime_add_software_perf_event(0, 0, 0);
		fds.push_back(fd);
		REQUIRE(fd >= 0);
		const epoll_data_t data{ .u64 = static_cast<uint64_t>(i + 1) };
		if (ringbuf) {
			REQUIRE(bpftime_add_ringbuf_fd_to_epoll(fd, epoll_fd,
								data) == 0);
			if (i < test.ready) {
				auto *payload = bpftime_ringbuf_reserve(
					fd, sizeof(uint64_t));
				REQUIRE(payload != nullptr);
				*static_cast<uint64_t *>(payload) = 42;
				bpftime_ringbuf_submit(fd, payload, 0);
			}
		} else {
			REQUIRE(bpftime_get_software_perf_event_raw_buffer(
					fd, 2 * getpagesize()) != nullptr);
			REQUIRE(bpftime_add_software_perf_event_fd_to_epoll(
					fd, epoll_fd, data) == 0);
			if (i < test.ready) {
				const uint64_t payload = 42;
				REQUIRE(bpftime_perf_event_output(
						fd, &payload,
						sizeof(payload)) == 0);
			}
		}
	}
	// Waiting must not consume data: unread sources remain ready next time.
	for (int repeat = 0; repeat < 2; ++repeat) {
		CAPTURE(repeat);
		std::array<epoll_event, 5> events;
		for (auto &event : events)
			event = { .events = EPOLLOUT, .data = { .u64 = 99 } };
		REQUIRE(pthread_sigmask(SIG_SETMASK, &before, nullptr) == 0);
		const int count = bpftime_epoll_wait(
			epoll_fd, events.data(), test.capacity, test.timeout);
		sigset_t after;
		const int mask_result =
			pthread_sigmask(SIG_SETMASK, nullptr, &after);
		// Restore immediately, including when testing the broken
		// implementation.
		REQUIRE(pthread_sigmask(SIG_SETMASK, &original, nullptr) == 0);
		REQUIRE(mask_result == 0);
		CHECK(count == test.expected);
		for (int signal = 1; signal < NSIG; ++signal) {
			CAPTURE(signal);
			CHECK(sigismember(&after, signal) ==
			      sigismember(&before, signal));
		}
		REQUIRE(count >= 0);
		REQUIRE(count <= test.capacity);
		bool seen[3] = {};
		for (int i = 0; i < count; ++i) {
			CHECK(events[i].events == EPOLLIN);
			const auto id = events[i].data.u64;
			REQUIRE(id >= 1);
			REQUIRE(id <= static_cast<uint64_t>(test.ready));
			CHECK_FALSE(seen[id]);
			seen[id] = true;
		}
		for (size_t i = test.capacity; i < events.size(); ++i) {
			CHECK(events[i].events == EPOLLOUT);
			CHECK(events[i].data.u64 == 99);
		}
	}
}

static volatile std::sig_atomic_t epoll_signal_test_seen = 0;
static void epoll_signal_test_handler(int)
{
	epoll_signal_test_seen = 1;
}

TEST_CASE("Epoll honors a blocked pending signal while a source stays ready",
	  "[epoll]")
{
	sigset_t original;
	REQUIRE(pthread_sigmask(SIG_SETMASK, nullptr, &original) == 0);
	const char *old_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	const bool had_name = old_name != nullptr;
	const std::string previous_name = old_name ? old_name : "";
	const std::string name = "EpollSignalShm-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	std::vector<int> fds;
	bool initialized = false;
	BOOST_SCOPE_EXIT_ALL(&)
	{
		// Drain while the signal is still blocked so a leftover pending
		// SIGINT (e.g. from the pre-fix implementation) cannot fire and
		// terminate the test process on the next unblock.
		sigset_t drain;
		sigemptyset(&drain);
		sigaddset(&drain, SIGINT);
		const timespec zero{ .tv_sec = 0, .tv_nsec = 0 };
		siginfo_t drained;
		while (sigtimedwait(&drain, &drained, &zero) >= 0) {
		}
		pthread_sigmask(SIG_SETMASK, &original, nullptr);
		struct sigaction reset;
		std::memset(&reset, 0, sizeof(reset));
		reset.sa_handler = SIG_DFL;
		sigaction(SIGINT, &reset, nullptr);
		if (initialized) {
			for (int fd : fds) {
				if (fd >= 0) {
					bpftime_close(fd);
					close(fd);
				}
			}
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
		if (had_name)
			setenv("BPFTIME_GLOBAL_SHM_NAME", previous_name.c_str(),
			       1);
		else
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	};

	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1) == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	initialized = true;

	const int epoll_fd = bpftime_epoll_create();
	fds.push_back(epoll_fd);
	REQUIRE(epoll_fd >= 0);

	// A software perf event that always has data forces the ready
	// (next_id > 0) path on every iteration, mirroring the high-frequency
	// tracepoint that previously starved signal delivery.
	const int evt_fd = bpftime_add_software_perf_event(0, 0, 0);
	fds.push_back(evt_fd);
	REQUIRE(evt_fd >= 0);
	REQUIRE(bpftime_get_software_perf_event_raw_buffer(
			evt_fd, 2 * getpagesize()) != nullptr);
	const epoll_data_t data{ .u64 = 1 };
	REQUIRE(bpftime_add_software_perf_event_fd_to_epoll(evt_fd, epoll_fd,
							    data) == 0);
	const uint64_t payload = 42;
	REQUIRE(bpftime_perf_event_output(evt_fd, &payload,
					  sizeof(payload)) == 0);

	// Block SIGINT, install an observable handler, then make SIGINT
	// pending so it stays queued instead of being delivered immediately.
	sigset_t block;
	sigemptyset(&block);
	sigaddset(&block, SIGINT);
	REQUIRE(pthread_sigmask(SIG_BLOCK, &block, nullptr) == 0);
	struct sigaction sa;
	std::memset(&sa, 0, sizeof(sa));
	sa.sa_handler = epoll_signal_test_handler;
	REQUIRE(sigaction(SIGINT, &sa, nullptr) == 0);
	epoll_signal_test_seen = 0;
	REQUIRE(raise(SIGINT) == 0);
	sigset_t pending;
	REQUIRE(sigpending(&pending) == 0);
	REQUIRE(sigismember(&pending, SIGINT) == 1);

	// The wait must be interrupted by the pending signal even though a
	// source is ready: return EINTR, run the handler, and consume the
	// signal rather than silently handing back events.
	errno = 0;
	std::array<epoll_event, 4> events;
	const int count = bpftime_epoll_wait(epoll_fd, events.data(),
					     events.size(), 1000);
	CHECK(count == -1);
	CHECK(errno == EINTR);
	CHECK(epoll_signal_test_seen == 1);
	sigset_t after;
	REQUIRE(sigpending(&after) == 0);
	CHECK(sigismember(&after, SIGINT) == 0);
}
#endif
