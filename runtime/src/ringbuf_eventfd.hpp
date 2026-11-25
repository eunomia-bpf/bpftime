/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _RINGBUF_EVENTFD_HPP
#define _RINGBUF_EVENTFD_HPP

#if defined(__linux__)
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace bpftime
{

// Server-side exporter of ringbuf kernel eventfds.
//
// The syscall-server process creates one kernel eventfd per
// BPF_MAP_TYPE_RINGBUF. Producer processes (injected agents) cannot inherit
// that fd: relying on /proc/<pid>/fd or pidfd_getfd would require
// ptrace/capabilities. Instead the server binds one abstract AF_UNIX
// SOCK_STREAM socket per shared-memory segment and hands a dup of the requested
// eventfd to any connected peer via SCM_RIGHTS. AF_UNIX + SCM_RIGHTS need no
// capabilities and work under CAP_DROP=ALL, no_new_privs, default Docker/K8s
// seccomp and yama ptrace_scope=3.
//
// A received dup only allows waking the ringbuf consumer; ring contents stay
// in shared memory, so the socket exposes no trace data.
class ringbuf_eventfd_exporter {
    public:
	// lookup_creator_eventfd maps a ringbuf map fd to the eventfd fd owned
	// by this process, returning -1 for unknown/non-ringbuf fds.
	ringbuf_eventfd_exporter(std::string shm_name,
				 std::function<int(int)> lookup_creator_eventfd);
	~ringbuf_eventfd_exporter();
	ringbuf_eventfd_exporter(const ringbuf_eventfd_exporter &) = delete;
	ringbuf_eventfd_exporter &
	operator=(const ringbuf_eventfd_exporter &) = delete;

	bool start();
	void stop();

    private:
	void run();
	int listen_fd = -1;
	std::string shm_name;
	std::function<int(int)> lookup_creator_eventfd;
	std::thread worker;
};

} // namespace bpftime

extern "C" {
// Client side: return a process-local, cached eventfd fd for the ringbuf map
// fd. The creator process gets its own kernel fd back; other processes fetch a
// SCM_RIGHTS dup from the exporter. Returns -1 when no eventfd is available,
// in which case callers skip the wakeup (correctness is preserved by the
// epoll_wait pre-block data check).
int bpftime_ringbuf_get_local_eventfd(int ringbuf_map_fd);
// Close and forget a cached dup. The creator's own fd is never closed here.
void bpftime_ringbuf_drop_local_eventfd(int ringbuf_map_fd);
// Read and discard one eventfd counter value (non-blocking). Returns 0 when a
// value was drained, EAGAIN when the eventfd was already empty.
int bpftime_ringbuf_drain_local_eventfd(int ringbuf_map_fd);
}

#endif // __linux__
#endif // _RINGBUF_EVENTFD_HPP
