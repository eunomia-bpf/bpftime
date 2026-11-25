/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#if defined(__linux__)
#include "ringbuf_eventfd.hpp"
#include "bpftime_shm.hpp"
#include "handler/handler_manager.hpp"
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <mutex>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <utility>

namespace
{
// Abstract socket address: a leading NUL in sun_path selects the abstract
// namespace, so nothing is created in the filesystem and no directory
// permission is involved.
struct mapfd_addr {
	struct sockaddr_un addr;
	socklen_t len;
};

mapfd_addr make_mapfd_addr(const std::string &shm_name)
{
	constexpr char prefix[] = "\0bpftime-mapfd-"; // includes leading NUL
	constexpr size_t prefix_len = sizeof(prefix) - 1;
	mapfd_addr out{};
	out.addr.sun_family = AF_UNIX;
	size_t cap = sizeof(out.addr.sun_path) - 1;
	size_t name_len = shm_name.size();
	if (prefix_len + name_len > cap)
		name_len = cap - prefix_len;
	std::memcpy(out.addr.sun_path, prefix, prefix_len);
	std::memcpy(out.addr.sun_path + prefix_len, shm_name.data(), name_len);
	out.len = static_cast<socklen_t>(
		offsetof(struct sockaddr_un, sun_path) + prefix_len + name_len);
	return out;
}

bool write_all(int fd, const void *buf, size_t len)
{
	const auto *p = static_cast<const uint8_t *>(buf);
	while (len > 0) {
		ssize_t n = ::write(fd, p, len);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			return false;
		}
		p += n;
		len -= static_cast<size_t>(n);
	}
	return true;
}

bool read_full(int fd, void *buf, size_t len)
{
	auto *p = static_cast<uint8_t *>(buf);
	while (len > 0) {
		ssize_t n = ::read(fd, p, len);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			return false;
		}
		if (n == 0)
			return false;
		p += n;
		len -= static_cast<size_t>(n);
	}
	return true;
}

void send_eventfd(int client, int eventfd)
{
	char control[CMSG_SPACE(sizeof(int))];
	std::memset(control, 0, sizeof(control));
	char payload = 'x';
	struct iovec io{ .iov_base = &payload, .iov_len = sizeof(payload) };
	struct msghdr msg{};
	msg.msg_iov = &io;
	msg.msg_iovlen = 1;
	msg.msg_control = control;
	msg.msg_controllen = sizeof(control);
	struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SCM_RIGHTS;
	cmsg->cmsg_len = CMSG_LEN(sizeof(int));
	std::memcpy(CMSG_DATA(cmsg), &eventfd, sizeof(int));
	// SCM_RIGHTS transfers a dup; the server keeps owning its original fd.
	if (sendmsg(client, &msg, 0) < 0)
		SPDLOG_DEBUG("ringbuf eventfd export sendmsg failed: {}",
			     errno);
}

int receive_eventfd(int sock)
{
	char control[CMSG_SPACE(sizeof(int))];
	std::memset(control, 0, sizeof(control));
	char payload = 0;
	struct iovec io{ .iov_base = &payload, .iov_len = sizeof(payload) };
	struct msghdr msg{};
	msg.msg_iov = &io;
	msg.msg_iovlen = 1;
	msg.msg_control = control;
	msg.msg_controllen = sizeof(control);
	if (recvmsg(sock, &msg, 0) < 0)
		return -1;
	for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c != nullptr;
	     c = CMSG_NXTHDR(&msg, c)) {
		if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
			int fd;
			std::memcpy(&fd, CMSG_DATA(c), sizeof(fd));
			int flags = fcntl(fd, F_GETFD);
			if (flags >= 0)
				fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
			return fd;
		}
	}
	errno = ENOENT;
	return -1;
}

int request_eventfd_dup(int map_fd, const std::string &shm_name)
{
	int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (sock < 0)
		return -1;
	auto [addr, len] = make_mapfd_addr(shm_name);
	int rc = -1;
	if (connect(sock, reinterpret_cast<struct sockaddr *>(&addr), len) ==
		    0 &&
	    write_all(sock, &map_fd, sizeof(map_fd))) {
		rc = receive_eventfd(sock);
	}
	close(sock);
	return rc;
}

struct cached_entry {
	int eventfd;
	bool dup_owned;
};

std::mutex &local_eventfd_cache_mutex()
{
	static std::mutex mutex;
	return mutex;
}

std::unordered_map<int, cached_entry> &local_eventfd_cache()
{
	static std::unordered_map<int, cached_entry> cache;
	return cache;
}
} // namespace

namespace bpftime
{
ringbuf_eventfd_exporter::ringbuf_eventfd_exporter(
	std::string shm_name, std::function<int(int)> lookup_creator_eventfd)
	: shm_name(std::move(shm_name)),
	  lookup_creator_eventfd(std::move(lookup_creator_eventfd))
{
}

ringbuf_eventfd_exporter::~ringbuf_eventfd_exporter()
{
	stop();
}

bool ringbuf_eventfd_exporter::start()
{
	listen_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (listen_fd < 0) {
		SPDLOG_WARN("ringbuf eventfd exporter socket failed: {}",
			    errno);
		return false;
	}
	auto [addr, len] = make_mapfd_addr(shm_name);
	if (bind(listen_fd, reinterpret_cast<struct sockaddr *>(&addr), len) <
		    0 ||
	    listen(listen_fd, 64) < 0) {
		SPDLOG_WARN(
			"ringbuf eventfd exporter bind/listen failed (shm={}): {}",
			shm_name, errno);
		close(listen_fd);
		listen_fd = -1;
		return false;
	}
	worker = std::thread([this] { run(); });
	return true;
}

void ringbuf_eventfd_exporter::stop()
{
	if (listen_fd >= 0) {
		// Abstract sockets disappear with the fd; unblock accept().
		shutdown(listen_fd, SHUT_RDWR);
		int fd = listen_fd;
		listen_fd = -1;
		close(fd);
	}
	if (worker.joinable())
		worker.join();
}

void ringbuf_eventfd_exporter::run()
{
	for (;;) {
		int client = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
		if (client < 0) {
			if (errno == EINTR)
				continue;
			return;
		}
		int32_t map_fd = -1;
		if (read_full(client, &map_fd, sizeof(map_fd))) {
			int eventfd = lookup_creator_eventfd(map_fd);
			if (eventfd >= 0)
				send_eventfd(client, eventfd);
		}
		close(client);
	}
}
} // namespace bpftime

extern "C" int bpftime_ringbuf_get_local_eventfd(int ringbuf_map_fd)
{
	std::lock_guard<std::mutex> guard(local_eventfd_cache_mutex());
	auto &cache = local_eventfd_cache();
	if (auto it = cache.find(ringbuf_map_fd); it != cache.end())
		return it->second.eventfd;

	int owner_pid = bpftime_ringbuf_eventfd_owner_pid(ringbuf_map_fd);
	if (owner_pid <= 0)
		return -1;

	int eventfd;
	bool dup_owned;
	if (owner_pid == getpid()) {
		// The creating process uses its original kernel fd directly.
		eventfd = bpftime_ringbuf_creator_eventfd(ringbuf_map_fd);
		dup_owned = false;
	} else {
		eventfd = request_eventfd_dup(ringbuf_map_fd,
					      bpftime::get_global_shm_name());
		dup_owned = true;
	}
	if (eventfd < 0)
		return -1;
	cache.emplace(ringbuf_map_fd, cached_entry{ .eventfd = eventfd,
						    .dup_owned = dup_owned });
	return eventfd;
}

extern "C" void bpftime_ringbuf_drop_local_eventfd(int ringbuf_map_fd)
{
	std::lock_guard<std::mutex> guard(local_eventfd_cache_mutex());
	auto &cache = local_eventfd_cache();
	auto it = cache.find(ringbuf_map_fd);
	if (it == cache.end())
		return;
	if (it->second.dup_owned)
		close(it->second.eventfd);
	cache.erase(it);
}

extern "C" int bpftime_ringbuf_drain_local_eventfd(int ringbuf_map_fd)
{
	int eventfd = bpftime_ringbuf_get_local_eventfd(ringbuf_map_fd);
	if (eventfd < 0)
		return EAGAIN;
	uint64_t value = 0;
	ssize_t rc = read(eventfd, &value, sizeof(value));
	if (rc == (ssize_t)sizeof(value))
		return 0;
	return errno != 0 ? errno : EIO;
}
#endif // __linux__
