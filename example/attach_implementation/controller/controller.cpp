
#include "bpftime_shm.hpp"
#include "request_filter.h"
#include "spdlog/spdlog.h"
#include <bpftime_prog.hpp>
#include <bpftime_object.hpp>
#include <cstddef>
#include <signal.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <vector>
static const int NGINX_REQUEST_FILTER_ATTACH_TYPE = 2001;
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static const char *ebpf_prog_path = TOSTRING(EBPF_PROGRAM_PATH);

using namespace bpftime;

static bool stop = false;
static void sig_handler(int sig)
{
	stop = true;
}

static int handle_data(void *, void *buf, size_t sz)
{
	request_filter_event *evt = (request_filter_event *)buf;
	// SPDLOG_INFO("{} {}", evt->url, evt->accepted);
	if (evt->accepted) {
		SPDLOG_INFO("Accepted: {}", evt->url);
	} else {
		SPDLOG_INFO("Rejected: {}", evt->url);
	}
	return 0;
}

int main(int argc, const char **argv)
{
	if (argc != 2) {
		SPDLOG_ERROR("Usage: {} [URL prefix to accept]", argv[0]);
		return 1;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	SPDLOG_INFO("eBPF program at {}", ebpf_prog_path);
	std::unique_ptr<bpftime_object, decltype(&bpftime_object_close)> obj(
		bpftime_object_open(ebpf_prog_path), &bpftime_object_close);
	auto prog = bpftime_object__next_program(obj.get(), nullptr);
	int rb_map_fd = bpftime_maps_create(
		-1, "nginx_ringbuf",
		bpf_map_attr{ .type = (int)bpf_map_type::BPF_MAP_TYPE_RINGBUF,
			      .max_ents = 1024 * 256 });
	SPDLOG_INFO("Ringbuf map fd is {}", rb_map_fd);
	auto insns = prog->get_insns();
	for (auto &insn : insns) {
		if (insn.code == 0x18 && insn.src_reg == 0) {
			insn.src_reg = 1;
			insn.imm = rb_map_fd;
			SPDLOG_INFO("Patched first instruction");
			break;
		}
	}
	int pfd =
		bpftime_progs_create(-1, insns.data(), insns.size(),
				     "nginx_request_filter",
				     (int)bpf_prog_type::BPF_PROG_TYPE_KPROBE);
	SPDLOG_INFO("Program fd is {}", pfd);
	int perf_event_fd = bpftime_add_custom_perf_event(
		NGINX_REQUEST_FILTER_ATTACH_TYPE, argv[1]);
	SPDLOG_INFO("Perf event fd is {}", perf_event_fd);
	int link_fd = bpftime_attach_perf_to_bpf(perf_event_fd, pfd);
	SPDLOG_INFO("Link fd is {}", link_fd);
	int epoll_fd = bpftime_epoll_create();
	SPDLOG_INFO("Epoll fd is {}", epoll_fd);
	if (int err = bpftime_add_ringbuf_fd_to_epoll(
		    rb_map_fd, epoll_fd, epoll_data_t{ .fd = rb_map_fd });
	    err < 0) {
		SPDLOG_ERROR("Unable to add ringbuf fd to epoll: {}", err);
		return 1;
	}
	while (!stop) {
		epoll_event out;
		SPDLOG_DEBUG("Polling");
		int ret = bpftime_epoll_wait(epoll_fd, &out, 1, 1000);
		if (ret == 1) {
			SPDLOG_DEBUG("Received data");
			if (int err = bpftime_poll_from_ringbuf(
				    out.data.fd, nullptr, handle_data);
			    err < 0) {
				SPDLOG_ERROR("Unable to poll from ringbuf: {}",
					     err);
			}
		}
	}
	return 0;
}
