/* SPDX-License-Identifier: MIT
 *
 * bpftime-remote-host: platform-neutral target daemon. Owns the local Boost
 * shared-memory segment, serves the wire protocol's SHM/RPC operations, and
 * streams ringbuf / software-perf-event samples to the control plane. The
 * OS-specific proc/spawn/ELF work lives in the platform_target backend.
 */
#ifndef BPFTIME_REMOTE_HOST_HPP
#define BPFTIME_REMOTE_HOST_HPP
#include <json.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace bpftime
{
namespace remote
{

struct SessionResources {
	std::vector<int> links;
	std::vector<int> perfs;
	std::vector<int> progs;
	std::vector<int> maps;
	std::vector<pid_t> attached_pids;
};

class RemoteHost {
    public:
	explicit RemoteHost(std::string agent_lib_path);
	~RemoteHost();

	void initialize_shm();
	void shutdown_shm();

	// Dispatch one authenticated request frame; response blob (if any) is
	// appended to resp_blob.
	nlohmann::json dispatch(const nlohmann::json &req,
				const std::vector<uint8_t> &blob,
				std::vector<uint8_t> &resp_blob);

	// Reclaim SHM objects allocated by a finished client session so
	// back-to-back tool runs do not accumulate duplicate probes.
	void cleanup_session(const SessionResources &session);

	// subscribe_events loop: pushes event frames on fd until stop is set.
	void run_event_stream(int client_fd, std::atomic<bool> &stop);

	const std::string &shm_name() const
	{
		return shm_name_;
	}

    private:
	nlohmann::json handle_hello();
	nlohmann::json handle_proc_check(const nlohmann::json &req);
	nlohmann::json handle_proc_maps(const nlohmann::json &req);
	nlohmann::json handle_fetch_elf(const nlohmann::json &req,
					std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_spawn(const nlohmann::json &req);
	nlohmann::json handle_resume(const nlohmann::json &req);

	nlohmann::json handle_map_create(const nlohmann::json &req);
	nlohmann::json handle_map_lookup(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob,
					 std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_map_update(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob);
	nlohmann::json handle_map_delete(const nlohmann::json &req,
					 const std::vector<uint8_t> &blob);
	nlohmann::json handle_map_get_next_key(
		const nlohmann::json &req, const std::vector<uint8_t> &blob,
		std::vector<uint8_t> &resp_blob);
	nlohmann::json handle_prog_load(const nlohmann::json &req,
				       const std::vector<uint8_t> &blob);
	nlohmann::json handle_perf_event_open(const nlohmann::json &req);
	nlohmann::json handle_bpf_link_create(const nlohmann::json &req);
	nlohmann::json handle_perf_state(const nlohmann::json &req,
					 bool enable);

	std::string agent_lib_path_;
	std::string shm_name_;
	std::mutex shm_op_mutex_;
};

nlohmann::json error_response(int err, const std::string &msg);

} // namespace remote
} // namespace bpftime

#endif
