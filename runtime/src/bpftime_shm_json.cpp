#include "bpftime_shm.hpp"
#include "handler/epoll_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_shm_internal.hpp>
#include <cstdio>
#include <sys/epoll.h>
#include <unistd.h>
#include <variant>
#include <fstream>
#include <json.hpp>

using namespace bpftime;
using json = nlohmann::json;
using namespace std;

static inline std::string bufferToHexString(const unsigned char* buffer, size_t bufferSize)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < bufferSize; i++)
    {
        ss << std::setw(2) << static_cast<int>(buffer[i]);
    }
    return ss.str();
}

static inline void hexStringToBuffer(const std::string& hexString, unsigned char* buffer, size_t bufferSize)
{
    if (hexString.length() != bufferSize * 2)
    {
        std::cerr << "Invalid hex string length." << std::endl;
        return;
    }

    for (size_t i = 0; i < bufferSize; i++)
    {
        std::string byteString = hexString.substr(i * 2, 2);
        buffer[i] = static_cast<unsigned char>(std::stoi(byteString, nullptr, 16));
    }
}

extern "C" void bpftime_import_global_shm_from_json(const char *filename)
{
}

extern "C" void bpftime_export_global_shm_to_json(const char *filename)
{
	std::ofstream file(filename);
	json j;

	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
        spdlog::error("No manager found in the shared memory");
		return;
	}
    for (std::size_t i = 0; i < manager->size(); i++) {
		// skip uninitialized handlers
		if (!manager->is_allocated(i)) {
			continue;
		}
		auto &handler = manager->get_handler(i);
		// load the bpf prog
		if (std::holds_alternative<bpf_prog_handler>(handler)) {
			auto &prog_handler =
				std::get<bpf_prog_handler>(handler);
			const ebpf_inst *insns = prog_handler.insns.data();
			size_t cnt = prog_handler.insns.size();
			const char *name = prog_handler.name.c_str();
			// record the prog into json, key is the index of the prog
            j[std::to_string(i)] = { { "insns", insns },
                         { "cnt", cnt },
                         { "name", name } };
			spdlog::info("find prog fd={} name={}", i,
				      prog_handler.name);
		} else if (std::holds_alternative<bpf_map_handler>(handler)) {
			spdlog::debug("bpf_map_handler found at {}", i);
		} else if (std::holds_alternative<bpf_perf_event_handler>(
				   handler)) {
			spdlog::debug("Will handle bpf_perf_events later...");

		} else if (std::holds_alternative<epoll_handler>(handler)) {
			spdlog::debug(
				"No extra operations needed for epoll_handler..");
		} else {
			spdlog::error("Unsupported handler type {}",
				      handler.index());
			return -1;
		}
	}
}
