/* SPDX-License-Identifier: MIT
 *
 * bpftime-darwin-host: host-side bridge daemon. Owns the local Boost shared
 * memory segment, serves the wire protocol, synthesizes stub ELF files for
 * Mach-O images, spawns workloads and streams ringbuf / perf-event samples
 * to the control plane.
 */
#ifndef BPFTIME_DARWIN_HOST_HPP
#define BPFTIME_DARWIN_HOST_HPP
#include "frame.hpp"
#include "macho_reader.hpp"
#include <json.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bpftime
{
namespace darwin
{

struct SpawnState {
	int spawn_id = 0;
	pid_t pid = -1;
	bool held = false; // started with POSIX_SPAWN_START_SUSPENDED
	int out_read_fd = -1;
	int err_read_fd = -1;
	std::mutex output_mutex;
	std::vector<uint8_t> captured_output;
	std::atomic<bool> exited{ false };
	std::atomic<int> exit_code{ 0 };
	std::atomic<int> term_sig{ 0 };
};

struct SessionResources {
	std::vector<int> links;
	std::vector<int> perfs;
	std::vector<int> progs;
	std::vector<int> maps;
	std::vector<pid_t> attached_pids;
};

class DarwinHost {
    public:
	explicit DarwinHost(std::string agent_dylib_path);
	~DarwinHost();

	void initialize_shm();
	void shutdown_shm();

	// Dispatch one request frame; the response binary blob (if any) is
	// appended to resp_blob.
	nlohmann::json dispatch(const nlohmann::json &req,
				const std::vector<uint8_t> &blob,
				std::vector<uint8_t> &resp_blob);

	// Reclaim SHM objects of a finished client session so back-to-back
	// tool runs do not accumulate duplicate probes.
	void cleanup_session(const SessionResources &session);

	// subscribe_events loop: send frames until stop is set.
	void run_event_stream(int client_fd, std::atomic<bool> &stop);

	const std::string &shm_name() const
	{
		return shm_name_;
	}

    private:
	nlohmann::json handle_hello();
	nlohmann::json handle_proc_check(const nlohmann::json &req);
	nlohmann::json handle_proc_maps(const nlohmann::json &req,
					std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_resolve_lib(const nlohmann::json &req);
	nlohmann::json handle_synth_elf(const nlohmann::json &req,
					std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_map_create(const nlohmann::json &req);
	nlohmann::json handle_map_lookup(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob,
					 std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_map_update(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob);
	nlohmann::json
	handle_map_update_batch(const nlohmann::json &req,
				const std::vector<uint8_t> &blob);
	nlohmann::json handle_map_delete(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob);
	nlohmann::json handle_map_get_next_key(const nlohmann::json &req,
					       const std::vector<uint8_t> &blob,
					       std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_prog_load(const nlohmann::json &req,
					const std::vector<uint8_t> &blob);
	nlohmann::json handle_perf_event_open(const nlohmann::json &req);
	nlohmann::json handle_perf_attach(const nlohmann::json &req);
	nlohmann::json handle_perf_enable(const nlohmann::json &req);
	nlohmann::json handle_spawn(const nlohmann::json &req);
	nlohmann::json handle_cmd_ctl(const nlohmann::json &req,
				      std::vector<uint8_t> &resp_blob);

	void refresh_agent(pid_t pid);
	const ModuleSymbols &resolve_module_symbols(const std::string &path);
	void reap_spawn(SpawnState &st);

	std::string agent_dylib_path_;
	std::string shm_name_;
	std::mutex shm_op_mutex_;
	int next_spawn_id_ = 1;
	std::mutex spawn_mutex_;
	std::unordered_map<int, std::unique_ptr<SpawnState>> spawns_;

	std::mutex symbol_mutex_;
	// Parsed/enumerated module symbols keyed by request path (virtual
	// /bpftime-dyld paths keep their prefix).
	std::unordered_map<std::string, ModuleSymbols> symbol_cache_;
	std::unordered_map<std::string, UmbrellaIndex> umbrella_cache_;
};

nlohmann::json error_response(int err, const std::string &msg);

} // namespace darwin
} // namespace bpftime

#endif
