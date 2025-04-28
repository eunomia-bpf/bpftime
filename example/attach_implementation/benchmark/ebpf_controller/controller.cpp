#include "bpftime_shm.hpp"
#include "request_filter.h"
#include "spdlog/spdlog.h"
#include <bpftime_prog.hpp>
#include <bpftime_object.hpp>
#include <cstddef>
#include <signal.h>
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
	if (!prog) {
		SPDLOG_ERROR("Failed to get program at {}", ebpf_prog_path);
		return 1;
	}
	
	// Create counter map
	int counter_map_fd = bpftime_maps_create(
		-1, "request_counter",
		bpf_map_attr{ .type = (int)bpf_map_type::BPF_MAP_TYPE_ARRAY,
			      .key_size = sizeof(uint32_t),
			      .value_size = sizeof(request_counter),
			      .max_ents = 1 });
	SPDLOG_INFO("Counter map fd is {}", counter_map_fd);
	
	// Initialize counter with zeros
	uint32_t key = 0;
	request_counter counter = {0};
	int ret = bpftime_map_update_elem(counter_map_fd, &key, &counter, 0);
	if (ret != 0) {
		SPDLOG_ERROR("Failed to initialize counter map: {}", ret);
		return 1;
	}
	
	auto insns = prog->get_insns();
	for (auto &insn : insns) {
		if (insn.code == 0x18 && insn.src_reg == 0) {
			insn.src_reg = 1;
			insn.imm = counter_map_fd;
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
	
	// Previous counter values for calculating deltas
	request_counter prev_counter = {0};
	
	while (!stop) {
		// Sleep for a second
		sleep(1);
		
		// Poll counter map
		request_counter* curr_counter = (request_counter*)bpftime_map_lookup_elem(counter_map_fd, &key);
		if (!curr_counter) {
			SPDLOG_ERROR("Unable to lookup counter map");
			continue;
		}
		
		// Calculate and log deltas
		uint64_t accepted_delta = curr_counter->accepted_count - prev_counter.accepted_count;
		uint64_t rejected_delta = curr_counter->rejected_count - prev_counter.rejected_count;
		
		printf("Stats - Total: Accepted: %lu, Rejected: %lu\n", 
			curr_counter->accepted_count, curr_counter->rejected_count);
		printf("Stats - Last interval: Accepted: %lu, Rejected: %lu\n", 
			accepted_delta, rejected_delta);	
		
		// Update previous values for next iteration
		prev_counter = *curr_counter;
	}
	
	return 0;
}
