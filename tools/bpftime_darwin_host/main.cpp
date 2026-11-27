/* SPDX-License-Identifier: MIT
 *
 * bpftime-darwin-host entry point: TCP frame protocol server.
 */
#include "darwin_host.hpp"
#include "frame.hpp"

#include <arpa/inet.h>
#include <frida-gum.h>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

using namespace bpftime::darwin;
using namespace bpftime::darwin_bridge;

namespace
{
std::atomic<bool> g_stop{ false };

void handle_signal(int)
{
	g_stop.store(true);
}

std::string resolve_agent_dylib(char *argv0)
{
	const char *env = getenv("BPFTIME_AGENT_DYLIB");
	if (env && env[0])
		return env;
	char resolved[4096] = {};
	if (realpath(argv0, resolved) == nullptr)
		return "libbpftime-agent.dylib";
	std::string exe(resolved);
	size_t slash = exe.find_last_of('/');
	std::string dir =
		slash == std::string::npos ? "." : exe.substr(0, slash);
	// Build-tree layout:
	// <build>/tools/bpftime_darwin_host/bpftime-darwin-host
	return dir + "/../../runtime/agent/libbpftime-agent.dylib";
}

void serve_connection(int fd, DarwinHost &host)
{
	try {
		std::atomic<bool> stream_stop{ false };
		SessionResources session;
		auto guard = std::shared_ptr<void>(nullptr, [&](void *) {
			stream_stop.store(true);
			close(fd);
			try {
				host.cleanup_session(session);
			} catch (const std::exception &ex) {
				SPDLOG_WARN("session cleanup threw: {}",
					    ex.what());
			}
		});
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
			} catch (const std::exception &ex) {
				auto resp = error_response(
					EINVAL,
					std::string("bad json: ") + ex.what());
				send_frame(fd, resp.dump());
				continue;
			}
			std::string op = req.value("op", "");
			if (op == "subscribe_events") {
				host.run_event_stream(fd, stream_stop);
				return;
			}
			std::vector<uint8_t> resp_blob;
			nlohmann::json resp =
				host.dispatch(req, frame.blob, resp_blob);
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
						session.attached_pids.push_back(
							pid);
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
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("connection thread failed: {}", ex.what());
		close(fd);
	}
}
} // namespace

int main(int argc, char **argv)
{
	int port = 49512;
	if (const char *env = getenv("BPFTIME_DARWIN_PORT"))
		port = atoi(env);
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
			port = atoi(argv[++i]);
	}
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	signal(SIGPIPE, SIG_IGN);

	// Frida Gum powers dyld_shared_cache export enumeration.
	gum_init_embedded();

	std::string agent_path = resolve_agent_dylib(argv[0]);
	if (access(agent_path.c_str(), R_OK) != 0)
		SPDLOG_WARN("agent dylib not readable at {}", agent_path);

	DarwinHost host(agent_path);
	host.initialize_shm();

	int lfd = socket(AF_INET, SOCK_STREAM, 0);
	if (lfd < 0) {
		SPDLOG_ERROR("socket: {}", strerror(errno));
		return 1;
	}
	int yes = 1;
	setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
	setsockopt(lfd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons((uint16_t)port);
	if (bind(lfd, (sockaddr *)&addr, sizeof(addr)) != 0) {
		SPDLOG_ERROR("bind port {}: {}", port, strerror(errno));
		return 1;
	}
	if (listen(lfd, 16) != 0) {
		SPDLOG_ERROR("listen: {}", strerror(errno));
		return 1;
	}
	SPDLOG_INFO("bpftime-darwin-host listening on 0.0.0.0:{} (shm {})",
		    port, host.shm_name());

	while (!g_stop.load()) {
		sockaddr_in cli{};
		socklen_t clen = sizeof(cli);
		int cfd = accept(lfd, (sockaddr *)&cli, &clen);
		if (cfd < 0) {
			if (errno == EINTR)
				break;
			continue;
		}
		std::thread(serve_connection, cfd, std::ref(host)).detach();
	}
	close(lfd);
	SPDLOG_INFO("bpftime-darwin-host shutting down");
	return 0;
}
