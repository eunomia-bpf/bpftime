#pragma once
#ifdef __linux__
#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
namespace bpftime
{
inline void set_ipc_socket_timeout(int fd, int timeout_ms)
{
	timeval tv{ timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
	::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
	::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}
inline bool make_agent_ipc_addr(pid_t pid, struct sockaddr_un &addr,
				socklen_t &len)
{
	std::memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	std::string name = "bpftime-agent-" + std::to_string((int)pid);
	if (name.size() + 1 > sizeof(addr.sun_path))
		return false;
	std::memcpy(addr.sun_path + 1, name.data(), name.size());
	len = offsetof(struct sockaddr_un, sun_path) + 1 + name.size();
	return true;
}
inline bool try_send_agent_ipc(pid_t pid, std::string_view request,
			       std::string *response_out = nullptr,
			       int timeout_ms = 1000)
{
	int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return false;
	set_ipc_socket_timeout(fd, timeout_ms);
	struct sockaddr_un addr {};
	socklen_t len = 0;
	bool ok = make_agent_ipc_addr(pid, addr, len) &&
		  ::connect(fd, (struct sockaddr *)&addr, len) == 0;
	for (size_t off = 0; ok && off < request.size();) {
		ssize_t n = ::send(fd, request.data() + off,
				   request.size() - off, MSG_NOSIGNAL);
		if (n > 0)
			off += (size_t)n;
		else if (errno != EINTR)
			ok = false;
	}
	if (ok)
		(void)::shutdown(fd, SHUT_WR);
	if (ok && response_out != nullptr) {
		std::string resp;
		char rbuf[4096];
		for (;;) {
			ssize_t n = ::recv(fd, rbuf, sizeof(rbuf), 0);
			if (n == 0)
				break;
			if (n > 0)
				resp.append(rbuf, rbuf + n);
			else if (errno != EINTR) {
				ok = false;
				break;
			}
		}
		if (ok)
			*response_out = std::move(resp);
	}
	::close(fd);
	return ok;
}
} // namespace bpftime
#endif // __linux__
