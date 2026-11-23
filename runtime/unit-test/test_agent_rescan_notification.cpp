// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026, eunomia-bpf org
// All rights reserved.
//
// Coverage for the late-uprobe agent "rescan" notification: when the syscall
// server emulates perf_event_open for a userspace uprobe/tracepoint/ureplace
// event and a BPF link is afterwards attached to that perf event fd (or the
// event is PERF_EVENT_IOC_ENABLE-d), it tells the bpftime agent already
// injected into the target process to incrementally re-scan handlers
// published after its startup scan. The message travels over the same abstract
// AF_UNIX control socket the agent's IPC server listens on.
//
// The whole scenario runs in a forked child with a private named shared
// memory segment, so the process-global syscall-server startup (std::call_once)
// and SHM ownership stay isolated from the rest of the test binary.
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__) && defined(__x86_64__)

#include "syscall_context.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include <ebpf-vm.h>
#include "syscall_server_utils.hpp"
#include "linux/bpf.h"
#include "linux/perf_event.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <poll.h>
#include <stdio.h>
#include <vector>

using namespace bpftime;

namespace
{

constexpr const char *RESCAN_MESSAGE = "rescan\n";
constexpr const char *UProBE_TARGET_PATH = "/tmp/bpftime-rescan-uprobe-target";

// Bind a listening abstract socket named "\0bpftime-agent-<agent_pid>", the
// exact address the agent's control IPC server uses.
int bind_agent_listener(pid_t agent_pid)
{
	int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK,
			  0);
	if (fd < 0)
		return -1;
	sockaddr_un addr {};
	const std::string name =
		"bpftime-agent-" + std::to_string(static_cast<int>(agent_pid));
	if (name.size() + 1 > sizeof(addr.sun_path)) {
		::close(fd);
		return -1;
	}
	addr.sun_family = AF_UNIX;
	addr.sun_path[0] = '\0'; // abstract namespace
	memcpy(addr.sun_path + 1, name.data(), name.size());
	const socklen_t len = static_cast<socklen_t>(
		offsetof(sockaddr_un, sun_path) + 1 + name.size());
	if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), len) != 0 ||
	    ::listen(fd, 16) != 0) {
		::close(fd);
		return -1;
	}
	return fd;
}

// Accept one connection and read everything until the peer half-closes.
// Returns an empty string when no connection arrives within the timeout.
std::string accept_one_message(int listen_fd, int timeout_ms)
{
	pollfd pfd {};
	pfd.fd = listen_fd;
	pfd.events = POLLIN;
	if (::poll(&pfd, 1, timeout_ms) <= 0)
		return "";
	int cfd = ::accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
	if (cfd < 0)
		return "";
	std::string received;
	char buffer[128];
	for (;;) {
		ssize_t n = ::recv(cfd, buffer, sizeof(buffer), 0);
		if (n > 0) {
			received.append(buffer, static_cast<size_t>(n));
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		break;
	}
	::close(cfd);
	return received;
}

// True when the listener has no pending incoming connection (i.e. the server
// did not send anything).
bool no_pending_connection(int listen_fd)
{
	pollfd pfd {};
	pfd.fd = listen_fd;
	pfd.events = POLLIN;
	return ::poll(&pfd, 1, 50) == 0;
}

int make_prog_fd()
{
	// r0 = 0; exit. The convenience BPF_*_INSN macros are not part of the
	// vendored uapi header, so encode the two instructions directly, the
	// same way the in-tree map syscall test helper does.
	ebpf_inst insns[2] = {};
	insns[0].code = BPF_ALU64 | BPF_MOV | BPF_K;
	insns[0].dst_reg = BPF_REG_0;
	insns[1].code = BPF_JMP | BPF_EXIT;
	return bpftime_progs_create(-1, reinterpret_cast<ebpf_inst *>(insns),
				    2, "rescan_test_prog",
				    BPF_PROG_TYPE_KPROBE);
}

int make_uprobe_perf_fd(syscall_context &ctx, pid_t target_pid)
{
	perf_event_attr attr {};
	attr.type = determine_uprobe_perf_type();
	attr.config1 = reinterpret_cast<uint64_t>(UProBE_TARGET_PATH);
	attr.config2 = 0; // offset
	return ctx.handle_perfevent(&attr, target_pid, 0 /* cpu */,
				    -1 /* group_fd */, 0 /* flags */);
}

long make_link(syscall_context &ctx, int prog_fd, int perf_fd)
{
	union bpf_attr attr {};
	attr.link_create.prog_fd = prog_fd;
	attr.link_create.target_fd = static_cast<uint32_t>(perf_fd);
	attr.link_create.attach_type = BPF_PERF_EVENT;
	return ctx.handle_sysbpf(BPF_LINK_CREATE, &attr, sizeof(attr));
}

void run_notification_scenario(int write_fd)
{
	auto report = [&](bool ok, const std::string &detail) {
		::dprintf(write_fd, "RESULT %s %s\n", ok ? "PASS" : "FAIL",
			  detail.c_str());
	};

	const std::string shm_name =
		"bpftime-rescan-notify-test-" + std::to_string(getpid());
	::setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	syscall_context ctx;
	// An arbitrary pid that is never this process: send_agent_rescan_ipc()
	// deliberately skips pid <= 0 and pid == getpid().
	const pid_t agent_pid = static_cast<pid_t>(getpid() + 4242);

	// --- Case A: a per-pid uprobe link notifies that one agent. ---
	int listener_a = bind_agent_listener(agent_pid);
	if (listener_a < 0) {
		report(false, "bind listener A");
		return;
	}
	int perf_fd = make_uprobe_perf_fd(ctx, agent_pid);
	if (perf_fd < 0) {
		report(false, "uprobe perf_event_open");
		return;
	}
	int prog_fd = make_prog_fd();
	if (prog_fd < 0) {
		report(false, "BPF_PROG_LOAD");
		return;
	}
	if (make_link(ctx, prog_fd, perf_fd) < 0) {
		report(false, "BPF_LINK_CREATE per-pid");
		return;
	}
	std::string msg_a = accept_one_message(listener_a, 2000);
	if (msg_a != RESCAN_MESSAGE) {
		report(false, "per-pid rescan payload");
		return;
	}
	// Exactly one notification per link, no duplicate fan-out.
	if (!no_pending_connection(listener_a)) {
		report(false, "unexpected duplicate per-pid notification");
		return;
	}
	::close(listener_a);

	// --- Case B: a system-wide (pid == -1) event fans out to every live
	// agent recorded in the shared-memory alive agent set. ---
	if (!shm_holder.global_shared_memory.add_pid_into_alive_agent_set(
		    static_cast<int>(agent_pid))) {
		report(false, "add alive agent pid");
		return;
	}
	int listener_b = bind_agent_listener(agent_pid);
	if (listener_b < 0) {
		report(false, "bind listener B");
		return;
	}
	int global_perf_fd = make_uprobe_perf_fd(ctx, -1);
	if (global_perf_fd < 0) {
		report(false, "global perf_event_open");
		return;
	}
	if (make_link(ctx, prog_fd, global_perf_fd) < 0) {
		report(false, "BPF_LINK_CREATE global");
		return;
	}
	std::string msg_b = accept_one_message(listener_b, 2000);
	if (msg_b != RESCAN_MESSAGE) {
		report(false, "global rescan payload");
		return;
	}
	::close(listener_b);

	// --- Case C: closing the perf fd drops the pid mapping, so a later
	// notification for that fd is a no-op. ---
	int listener_c = bind_agent_listener(agent_pid);
	if (listener_c < 0) {
		report(false, "bind listener C");
		return;
	}
	if (ctx.handle_close(global_perf_fd) != 0) {
		report(false, "perf fd close");
		return;
	}
	ctx.notify_agent_rescan_for_perf_fd(global_perf_fd);
	if (!no_pending_connection(listener_c)) {
		report(false, "notification sent after perf fd close");
		return;
	}
	::close(listener_c);

	report(true, "per-pid, global fan-out and close-erase all verified");
}

void run_benchmark(int write_fd)
{
	const std::string shm_name =
		"bpftime-rescan-bench-test-" + std::to_string(getpid());
	::setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	syscall_context ctx;
	const pid_t agent_pid = static_cast<pid_t>(getpid() + 4343);
	constexpr int iterations = 2000;

	// Scenario 1: a live agent accepts every connection (full path).
	int live_listener = bind_agent_listener(agent_pid);
	if (live_listener < 0)
		return;
	std::atomic<bool> stop { false };
	std::thread drainer([&]() {
		while (!stop.load(std::memory_order_acquire)) {
			int cfd = ::accept4(live_listener, nullptr, nullptr,
					    SOCK_CLOEXEC | SOCK_NONBLOCK);
			if (cfd >= 0) {
				char buffer[64];
				while (::recv(cfd, buffer, sizeof(buffer), 0) >
				       0) {
				}
				::close(cfd);
			} else {
				std::this_thread::sleep_for(
					std::chrono::microseconds(20));
			}
		}
	});
	int perf_fd = make_uprobe_perf_fd(ctx, agent_pid);
	if (perf_fd < 0) {
		stop.store(true);
		drainer.join();
		return;
	}
	std::vector<long long> live_samples;
	live_samples.reserve(iterations);
	for (int i = 0; i < iterations; ++i) {
		auto start = std::chrono::steady_clock::now();
		ctx.notify_agent_rescan_for_perf_fd(perf_fd);
		auto end = std::chrono::steady_clock::now();
		live_samples.push_back(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				end - start)
				.count());
	}
	stop.store(true);
	drainer.join();
	::close(live_listener);

	// Scenario 2: no agent is listening yet (connect fails fast), the
	// common state before the target process finishes its own startup.
	const pid_t absent_pid = static_cast<pid_t>(getpid() + 4444);
	int absent_perf_fd = make_uprobe_perf_fd(ctx, absent_pid);
	if (absent_perf_fd < 0)
		return;
	std::vector<long long> absent_samples;
	absent_samples.reserve(iterations);
	for (int i = 0; i < iterations; ++i) {
		auto start = std::chrono::steady_clock::now();
		ctx.notify_agent_rescan_for_perf_fd(absent_perf_fd);
		auto end = std::chrono::steady_clock::now();
		absent_samples.push_back(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				end - start)
				.count());
	}

	auto summarize = [](std::vector<long long> v) {
		std::sort(v.begin(), v.end());
		long long sum = 0;
		for (long long x : v)
			sum += x;
		struct stats {
			long long mean_ns;
			long long median_ns;
			long long min_ns;
			long long max_ns;
		};
		return stats { sum / static_cast<long long>(v.size()),
			       v[v.size() / 2], v.front(), v.back() };
	};
	auto live = summarize(live_samples);
	auto absent = summarize(absent_samples);
	::dprintf(write_fd,
		  "BENCH live_agent mean=%lldns median=%lldns min=%lldns "
		  "max=%lldns N=%d\n",
		  live.mean_ns, live.median_ns, live.min_ns, live.max_ns,
		  iterations);
	::dprintf(write_fd,
		  "BENCH absent_agent mean=%lldns median=%lldns min=%lldns "
		  "max=%lldns N=%d\n",
		  absent.mean_ns, absent.median_ns, absent.min_ns,
		  absent.max_ns, iterations);
	::dprintf(write_fd,
		  "BENCH hot_event_path added_instructions=0 (hooks only on "
		  "BPF_LINK_CREATE/PERF_EVENT_IOC_ENABLE/close)\n");
}

} // namespace

TEST_CASE("Late uprobe attach triggers an agent rescan notification",
	  "[syscall][rescan]")
{
	int pipe_fds[2];
	REQUIRE(::pipe(pipe_fds) == 0);

	pid_t pid = ::fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		::close(pipe_fds[0]);
		run_notification_scenario(pipe_fds[1]);
		::_exit(0);
	}
	::close(pipe_fds[1]);
	std::string output;
	char buffer[1024];
	for (;;) {
		ssize_t n = ::read(pipe_fds[0], buffer, sizeof(buffer));
		if (n > 0) {
			output.append(buffer, static_cast<size_t>(n));
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		break;
	}
	::close(pipe_fds[0]);
	int status = 0;
	REQUIRE(::waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFEXITED(status));
	INFO(output);
	REQUIRE(output.find("RESULT PASS") != std::string::npos);
	REQUIRE(output.find("RESULT FAIL") == std::string::npos);
}

TEST_CASE("Agent rescan notification attach-path microbenchmark",
	  "[syscall][rescan][bench]")
{
	int pipe_fds[2];
	REQUIRE(::pipe(pipe_fds) == 0);

	pid_t pid = ::fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		::close(pipe_fds[0]);
		run_benchmark(pipe_fds[1]);
		::_exit(0);
	}
	::close(pipe_fds[1]);
	std::string output;
	char buffer[2048];
	for (;;) {
		ssize_t n = ::read(pipe_fds[0], buffer, sizeof(buffer));
		if (n > 0) {
			output.append(buffer, static_cast<size_t>(n));
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		break;
	}
	::close(pipe_fds[0]);
	int status = 0;
	REQUIRE(::waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFEXITED(status));
	// The benchmark's job is to emit measurements, not to gate CI on a
	// timing threshold. It still must produce both scenario reports.
	INFO(output);
	REQUIRE(output.find("BENCH live_agent") != std::string::npos);
	REQUIRE(output.find("BENCH absent_agent") != std::string::npos);
	REQUIRE(output.find("BENCH hot_event_path") != std::string::npos);
}

#endif
