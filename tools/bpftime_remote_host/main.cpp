/* SPDX-License-Identifier: MIT
 *
 * bpftime-remote-host entry point: authenticated TCP frame server.
 *
 * Security: binds 127.0.0.1 by default and requires a 256-bit shared token on
 * the very first frame of every connection (hello or subscribe_events).
 */
#include "frame.hpp"
#include "platform_target.hpp"
#include "remote_host.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

using namespace bpftime::remote;

namespace
{
std::atomic<bool> g_stop{ false };
// Write end of the self-pipe used to wake the poll() loop from a signal that
// may be delivered to any worker thread.
volatile sig_atomic_t g_sig_wr_fd = -1;

void handle_signal(int)
{
	g_stop.store(true);
	if (g_sig_wr_fd >= 0) {
		int saved = errno;
		(void)write(g_sig_wr_fd, "x", 1);
		errno = saved;
	}
}

constexpr size_t kMinTokenLen = 16;

// Length-folding constant-time comparison; never short-circuits on mismatch.
bool constant_time_equals(const std::string &a, const std::string &b)
{
	volatile uint8_t diff = (uint8_t)(a.size() ^ b.size());
	size_t n = std::min(a.size(), b.size());
	for (size_t i = 0; i < n; i++)
		diff |= (uint8_t)(a[i] ^ b[i]);
	return diff == 0;
}

std::string to_hex(const uint8_t *p, size_t n)
{
	static const char kHex[] = "0123456789abcdef";
	std::string out;
	out.resize(n * 2);
	for (size_t i = 0; i < n; i++) {
		out[2 * i] = kHex[p[i] >> 4];
		out[2 * i + 1] = kHex[p[i] & 0xf];
	}
	return out;
}

std::string default_token_path()
{
	const char *home = getenv("HOME");
	return std::string(home ? home : ".") + "/.bpftime/remote.token";
}

// Resolve the mandatory token: BPFTIME_REMOTE_TOKEN wins; otherwise read (or
// generate with 0600) the token file, defaulting to $HOME/.bpftime/remote.token.
std::string resolve_token(const std::string &token_file_arg)
{
	if (const char *env = getenv("BPFTIME_REMOTE_TOKEN");
	    env && std::strlen(env) >= kMinTokenLen)
		return env;

	std::string path = token_file_arg.empty() ? default_token_path() :
						    token_file_arg;
	std::ifstream in(path);
	if (in) {
		std::string tok((std::istreambuf_iterator<char>(in)),
				std::istreambuf_iterator<char>());
		while (!tok.empty() && (tok.back() == '\n' || tok.back() == '\r'))
			tok.pop_back();
		if (tok.size() >= kMinTokenLen)
			return tok;
	}

	uint8_t raw[32];
	int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		throw std::runtime_error("open /dev/urandom failed");
	size_t got = 0;
	while (got < sizeof(raw)) {
		ssize_t r = read(fd, raw + got, sizeof(raw) - got);
		if (r <= 0) {
			close(fd);
			throw std::runtime_error("read /dev/urandom failed");
		}
		got += (size_t)r;
	}
	close(fd);
	std::string tok = to_hex(raw, sizeof(raw));

	size_t slash = path.find_last_of('/');
	if (slash != std::string::npos) {
		std::string dir = path.substr(0, slash);
		// Best-effort: ignore EEXIST from a pre-existing directory.
		mkdir(dir.c_str(), 0700);
	}
	{
		std::ofstream out(path, std::ios::trunc);
		if (!out)
			throw std::runtime_error("cannot write token file " +
						 path);
		out << tok;
	}
	chmod(path.c_str(), 0600);
	SPDLOG_INFO("generated new remote token at {} (0600)", path);
	return tok;
}

void send_unauthorized(int fd)
{
	try {
		send_frame(fd,
			   nlohmann::json{ { "ok", false },
					   { "error", "unauthorized" } }
				   .dump());
	} catch (...) {
	}
}

void serve_connection(int fd, RemoteHost &host, const std::string &token)
{
	std::atomic<bool> stream_stop{ false };
	SessionResources session;
	auto guard = std::shared_ptr<void>(nullptr, [&](void *) {
		stream_stop.store(true);
		close(fd);
		try {
			host.cleanup_session(session);
		} catch (const std::exception &ex) {
			SPDLOG_WARN("session cleanup threw: {}", ex.what());
		}
	});

	bool authenticated = false;
	while (!g_stop.load()) {
		Frame frame;
		try {
			frame = recv_frame(fd);
		} catch (const std::exception &) {
			return;
		}
		nlohmann::json req;
		try {
			req = nlohmann::json::parse(frame.json);
		} catch (const std::exception &) {
			if (!authenticated)
				send_unauthorized(fd);
			return;
		}
		std::string op = req.value("op", "");

		if (!authenticated) {
			std::string presented = req.value("token", "");
			if ((op != "hello" && op != "subscribe_events") ||
			    presented.empty() ||
			    !constant_time_equals(presented, token)) {
				send_unauthorized(fd);
				return;
			}
			authenticated = true;
			if (op == "subscribe_events") {
				host.run_event_stream(fd, stream_stop);
				return;
			}
		}

		if (op == "subscribe_events") {
			host.run_event_stream(fd, stream_stop);
			return;
		}

		std::vector<uint8_t> resp_blob;
		nlohmann::json resp = host.dispatch(req, frame.blob,
						    resp_blob);
		if (resp.value("ok", false)) {
			if (resp.contains("host_link_id"))
				session.links.push_back(
					resp["host_link_id"].get<int>());
			if (resp.contains("host_perf_id"))
				session.perfs.push_back(
					resp["host_perf_id"].get<int>());
			if (resp.contains("host_prog_id"))
				session.progs.push_back(
					resp["host_prog_id"].get<int>());
			if (resp.contains("host_map_id"))
				session.maps.push_back(
					resp["host_map_id"].get<int>());
			if (op == "perf_event_enable") {
				pid_t pid = req.value("pid", -1);
				if (pid > 0)
					session.attached_pids.push_back(pid);
			}
		}
		try {
			send_frame(fd, resp.dump(),
				   resp_blob.empty() ? nullptr :
							   resp_blob.data(),
				   resp_blob.size());
		} catch (const std::exception &) {
			return;
		}
	}
}
} // namespace

int main(int argc, char **argv)
{
	int port = 49512;
	std::string bind_ip = "127.0.0.1";
	std::string token_file;
	bool allow_any = false;
	for (int i = 1; i < argc; i++) {
		if ((std::strcmp(argv[i], "-p") == 0 ||
		     std::strcmp(argv[i], "--port") == 0) &&
		    i + 1 < argc)
			port = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc)
			bind_ip = argv[++i];
		else if (std::strcmp(argv[i], "--token-file") == 0 &&
			 i + 1 < argc)
			token_file = argv[++i];
		else if (std::strcmp(argv[i], "--allow-any") == 0)
			allow_any = true;
	}
	if (const char *env = getenv("BPFTIME_REMOTE_PORT"))
		port = std::atoi(env);

	if (bind_ip == "0.0.0.0" && !allow_any) {
		SPDLOG_ERROR("refusing to bind 0.0.0.0 without --allow-any");
		return 1;
	}

	std::string token;
	try {
		token = resolve_token(token_file);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("token setup failed: {}", ex.what());
		return 1;
	}

	// Self-pipe so a signal delivered to any thread wakes the main poll().
	int sig_pipe[2];
	if (pipe(sig_pipe) != 0) {
		SPDLOG_ERROR("pipe: {}", strerror(errno));
		return 1;
	}
	for (int p : sig_pipe)
		fcntl(p, F_SETFD, FD_CLOEXEC);
	fcntl(sig_pipe[1], F_SETFL, O_NONBLOCK);
	g_sig_wr_fd = sig_pipe[1];
	struct sigaction sa {};
	sa.sa_handler = handle_signal;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);
	signal(SIGPIPE, SIG_IGN);

	platform_init();
	std::string agent_path = platform_default_agent_path(argv[0]);
	if (access(agent_path.c_str(), R_OK) != 0)
		SPDLOG_WARN("agent library not readable at {}", agent_path);

	RemoteHost host(agent_path);
	host.initialize_shm();

	int lfd = socket(AF_INET, SOCK_STREAM, 0);
	if (lfd < 0) {
		SPDLOG_ERROR("socket: {}", strerror(errno));
		return 1;
	}
	int yes = 1;
	setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef __APPLE__
	setsockopt(lfd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
	sockaddr_in addr {};
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
		SPDLOG_ERROR("invalid bind address {}", bind_ip);
		return 1;
	}
	if (bind(lfd, (sockaddr *)&addr, sizeof(addr)) != 0) {
		SPDLOG_ERROR("bind {}:{}: {}", bind_ip, port, strerror(errno));
		return 1;
	}
	if (listen(lfd, 16) != 0) {
		SPDLOG_ERROR("listen: {}", strerror(errno));
		return 1;
	}
	SPDLOG_INFO("bpftime-remote-host listening on {}:{} (shm {})",
		    bind_ip, port, host.shm_name());

	pollfd watch[2] = { { lfd, POLLIN, 0 },
			     { sig_pipe[0], POLLIN, 0 } };
	while (!g_stop.load()) {
		int pr = poll(watch, 2, -1);
		if (pr < 0) {
			if (errno == EINTR)
				continue;
			break;
		}
		if (watch[1].revents & (POLLIN | POLLHUP | POLLERR))
			break; // signaled
		if (!(watch[0].revents & POLLIN))
			continue;
		sockaddr_in cli {};
		socklen_t clen = sizeof(cli);
		int cfd = accept(lfd, (sockaddr *)&cli, &clen);
		if (cfd < 0) {
			if (errno == EINTR)
				continue;
			continue;
		}
		std::thread(serve_connection, cfd, std::ref(host),
			    std::cref(token))
			.detach();
	}
	close(lfd);
	close(sig_pipe[0]);
	close(sig_pipe[1]);
	SPDLOG_INFO("bpftime-remote-host shutting down");
	return 0;
}
