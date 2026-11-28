/* SPDX-License-Identifier: MIT
 *
 * bpftime-remote-host implementation: platform-neutral SHM/RPC operations and
 * the bounded event stream. OS-specific proc/spawn/ELF work is delegated to
 * the platform_target backend.
 */
#include "remote_host.hpp"
#include "frame.hpp"
#include "platform_target.hpp"

#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "ebpf-vm.h"
#include "handler/perf_event_handler.hpp"

#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>

extern char **environ;

namespace bpftime
{
namespace remote
{

namespace
{
constexpr int BPF_PSEUDO_MAP_FD_LOCAL = 1;
constexpr int BPF_PSEUDO_MAP_VALUE_LOCAL = 2;
constexpr uint8_t LDDW_OPCODE = 0x18;
constexpr int kMaxEventsPerPollSlice = 64;

uint32_t align8(uint32_t v)
{
	return (v + 7u) & ~7u;
}

bool is_percpu_type(int type)
{
	return type == 5 || type == 6; // BPF_MAP_TYPE_PERCPU_HASH / ARRAY
}

struct QueuedEvent {
	bool is_perf = false;
	int fd = -1;
	std::vector<uint8_t> payload;
};

struct RingCtx {
	int fd;
	std::atomic<bool> *stop;
	std::vector<QueuedEvent> *out;
	int count = 0;
};

int ringbuf_forward_cb(void *raw, void *data, size_t len)
{
	auto *c = static_cast<RingCtx *>(raw);
	if (c->stop->load())
		return -1;
	const auto *bytes = static_cast<const uint8_t *>(data);
	c->out->push_back(QueuedEvent{ false, c->fd,
				      std::vector<uint8_t>(bytes, bytes + len) });
	// Bound the lock hold time: commit the consumer position after 64.
	if (++c->count >= kMaxEventsPerPollSlice)
		return -1;
	return 0;
}
} // namespace

nlohmann::json error_response(int err, const std::string &msg)
{
	return nlohmann::json{ { "ok", false }, { "err", err },
			       { "msg", msg } };
}

RemoteHost::RemoteHost(std::string agent_lib_path)
	: agent_lib_path_(std::move(agent_lib_path)),
	  shm_name_("bpftime_remote_shm_" + std::to_string((uint64_t)getpid()))
{
}

RemoteHost::~RemoteHost()
{
	shutdown_shm();
}

void RemoteHost::initialize_shm()
{
	setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name_.c_str(), 1);
	// BCC creates a 256 MiB RINGBUF that bpftime double-backs; keep a
	// 1 GiB default so sessions never hit bad_alloc. Overridable via env.
	if (getenv("BPFTIME_SHM_MEMORY_MB") == nullptr)
		setenv("BPFTIME_SHM_MEMORY_MB", "1024", 1);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	SPDLOG_INFO("remote host: created SHM segment {}", shm_name_);
}

void RemoteHost::shutdown_shm()
{
	if (bpftime_global_shm_is_initialized()) {
		bpftime_destroy_global_shm();
		bpftime_remove_global_shm();
	}
}

nlohmann::json RemoteHost::handle_hello()
{
	PlatformIdentity id = platform_identity();
	return nlohmann::json{ { "ok", true }, { "os", id.os },
			       { "arch", id.arch },
			       { "num_cpus", id.num_cpus },
			       { "host_pid", (int)getpid() } };
}

nlohmann::json RemoteHost::handle_proc_check(const nlohmann::json &req)
{
	pid_t pid = req.value("pid", 0);
	ProcCheckResult r = platform_proc_check(pid);
	return nlohmann::json{ { "ok", true }, { "alive", r.alive },
			       { "comm", r.comm }, { "exe", r.exe },
			       { "status", r.status } };
}

nlohmann::json RemoteHost::handle_proc_maps(const nlohmann::json &req)
{
	pid_t pid = req.value("pid", 0);
	std::string maps = platform_proc_maps(pid);
	if (maps.empty())
		return error_response(ENOENT,
				      "no maps for pid " +
					      std::to_string(pid));
	return nlohmann::json{ { "ok", true }, { "maps", maps } };
}

nlohmann::json
RemoteHost::handle_fetch_elf(const nlohmann::json &req,
			     std::vector<uint8_t> &resp_blob)
{
	std::string path = req.value("path", "");
	if (path.empty())
		return error_response(EINVAL, "missing path");
	std::string canonical;
	resp_blob = platform_fetch_elf(path, canonical);
	if (resp_blob.empty())
		return error_response(ENOENT, "cannot read module " + path);
	return nlohmann::json{ { "ok", true },
			       { "canonical_path", canonical } };
}

nlohmann::json RemoteHost::handle_spawn(const nlohmann::json &req)
{
	auto argv_json = req.value("argv", nlohmann::json::array());
	if (argv_json.empty() || !argv_json[0].is_string())
		return error_response(EINVAL, "argv required");
	bool hold = req.value("hold", false);
	std::string cwd = req.value("cwd", "");

	std::vector<std::string> argv_storage;
	argv_storage.reserve(argv_json.size());
	for (const auto &a : argv_json)
		argv_storage.push_back(a.get<std::string>());
	std::vector<char *> cargv;
	cargv.reserve(argv_storage.size() + 1);
	for (auto &a : argv_storage)
		cargv.push_back(a.data());
	nlohmann::json empty_env = nlohmann::json::object();
	const nlohmann::json &extra_env =
		(req.contains("env") && req["env"].is_object()) ?
			      req["env"] :
			      empty_env;
	cargv.push_back(nullptr);

	// Build the complete environment in the parent so the post-fork child
	// only uses async-signal-safe calls (malloc after fork in a
	// multithreaded process can deadlock on another thread's allocator).
	std::vector<std::string> env_storage;
	env_storage.emplace_back(std::string(platform_preload_env_name()) +
				 "=" + agent_lib_path_);
	env_storage.emplace_back("BPFTIME_GLOBAL_SHM_NAME=" + shm_name_);
	env_storage.emplace_back("BPFTIME_ENABLE_AGENT_IPC=1");
	for (auto it = extra_env.begin(); it != extra_env.end(); ++it)
		env_storage.emplace_back(it.key() + "=" +
					 it.value().get<std::string>());
	if (environ != nullptr) {
		for (char **e = environ; *e != nullptr; ++e) {
			std::string entry(*e);
			size_t eq = entry.find('=');
			std::string key = eq == std::string::npos ? entry :
								     entry.substr(0, eq);
			if (key == platform_preload_env_name() ||
			    key == "BPFTIME_GLOBAL_SHM_NAME" ||
			    key == "BPFTIME_ENABLE_AGENT_IPC" ||
			    extra_env.contains(key))
				continue;
			env_storage.push_back(std::move(entry));
		}
	}
	std::vector<char *> cenvp;
	cenvp.reserve(env_storage.size() + 1);
	for (auto &e : env_storage)
		cenvp.push_back(e.data());
	cenvp.push_back(nullptr);

	try {
		pid_t pid = platform_spawn(cargv.data(), cenvp.data(), cwd,
					   hold);
		return nlohmann::json{ { "ok", true }, { "pid", (int)pid },
				       { "host_pid", (int)pid } };
	} catch (const std::exception &ex) {
		return error_response(errno ? errno : EIO, ex.what());
	}
}

nlohmann::json RemoteHost::handle_resume(const nlohmann::json &req)
{
	pid_t pid = req.value("pid", 0);
	if (pid <= 0)
		return error_response(EINVAL, "missing pid");
	platform_resume(pid);
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json RemoteHost::handle_map_create(const nlohmann::json &req)
{
	bpftime::bpf_map_attr attr;
	attr.type = req.value("map_type", 0);
	attr.key_size = req.value("key_size", 0u);
	attr.value_size = req.value("value_size", 0u);
	attr.max_ents = req.value("max_entries", 0u);
	attr.flags = req.value("flags", 0ull);
	std::string name = req.value("name", "");
	int id = bpftime_maps_create(-1, name.c_str(), attr);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "map_create failed");
	return nlohmann::json{ { "ok", true }, { "host_map_id", id } };
}

nlohmann::json
RemoteHost::handle_map_lookup(const nlohmann::json &req,
			      const std::vector<uint8_t> &blob,
			      std::vector<uint8_t> &resp_blob)
{
	int id = req.value("host_map_id", -1);
	bpftime::bpf_map_attr attr;
	const char *name = nullptr;
	bpftime::bpf_map_type type;
	if (bpftime_map_get_info(id, &attr, &name, &type) != 0)
		return error_response(ENOENT, "unknown map id");
	if (blob.size() < attr.key_size)
		return error_response(EINVAL, "key blob too short");
	const void *p = bpftime_map_lookup_elem(id, blob.data());
	if (p == nullptr)
		return error_response(ENOENT, "no such element");
	if (is_percpu_type(attr.type)) {
		int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
		uint32_t stride = align8(attr.value_size);
		resp_blob.resize((size_t)ncpu * stride);
		memcpy(resp_blob.data(), p, (size_t)ncpu * attr.value_size);
		return nlohmann::json{ { "ok", true },
				       { "is_percpu", true },
				       { "num_cpus", ncpu } };
	}
	resp_blob.assign((const uint8_t *)p,
			 (const uint8_t *)p + attr.value_size);
	return nlohmann::json{ { "ok", true }, { "is_percpu", false },
			       { "num_cpus", 1 } };
}

nlohmann::json
RemoteHost::handle_map_update(const nlohmann::json &req,
			      const std::vector<uint8_t> &blob)
{
	int id = req.value("host_map_id", -1);
	uint64_t flags = req.value("flags", 0ull);
	bpftime::bpf_map_attr attr;
	const char *name = nullptr;
	bpftime::bpf_map_type type;
	if (bpftime_map_get_info(id, &attr, &name, &type) != 0)
		return error_response(ENOENT, "unknown map id");
	size_t need = attr.key_size + attr.value_size;
	if (is_percpu_type(attr.type))
		need = attr.key_size +
		       (size_t)sysconf(_SC_NPROCESSORS_ONLN) *
			       align8(attr.value_size);
	if (blob.size() < need)
		return error_response(EINVAL, "blob too short");
	if (bpftime_map_update_elem(id, blob.data(),
				    blob.data() + attr.key_size, flags) != 0)
		return error_response(errno ? errno : EIO, "update failed");
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json
RemoteHost::handle_map_delete(const nlohmann::json &req,
			      const std::vector<uint8_t> &blob)
{
	int id = req.value("host_map_id", -1);
	bpftime::bpf_map_attr attr;
	const char *name = nullptr;
	bpftime::bpf_map_type type;
	if (bpftime_map_get_info(id, &attr, &name, &type) != 0)
		return error_response(ENOENT, "unknown map id");
	if (blob.size() < attr.key_size)
		return error_response(EINVAL, "key blob too short");
	if (bpftime_map_delete_elem(id, blob.data()) != 0)
		return error_response(errno ? errno : EIO, "delete failed");
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json
RemoteHost::handle_map_get_next_key(const nlohmann::json &req,
				    const std::vector<uint8_t> &blob,
				    std::vector<uint8_t> &resp_blob)
{
	int id = req.value("host_map_id", -1);
	bpftime::bpf_map_attr attr;
	const char *name = nullptr;
	bpftime::bpf_map_type type;
	if (bpftime_map_get_info(id, &attr, &name, &type) != 0)
		return error_response(ENOENT, "unknown map id");
	std::vector<uint8_t> next(attr.key_size, 0);
	const void *key = req.value("has_key", false) ? blob.data() : nullptr;
	if (key != nullptr && blob.size() < attr.key_size)
		return error_response(EINVAL, "key blob too short");
	int rc = bpftime_map_get_next_key(id, key, next.data());
	if (rc != 0) {
		if (errno == ENOENT)
			return nlohmann::json{ { "ok", true }, { "end", true } };
		return error_response(errno ? errno : EIO,
				      "get_next_key failed");
	}
	resp_blob = std::move(next);
	return nlohmann::json{ { "ok", true }, { "end", false } };
}

nlohmann::json
RemoteHost::handle_prog_load(const nlohmann::json &req,
			     const std::vector<uint8_t> &blob)
{
	if (blob.size() % sizeof(ebpf_inst) != 0)
		return error_response(EINVAL, "insn blob not 8-byte aligned");
	std::vector<ebpf_inst> insns(blob.size() / sizeof(ebpf_inst));
	memcpy(insns.data(), blob.data(), blob.size());

	std::unordered_map<int, int> fd_map;
	for (const auto &pair : req.value("fd_map", nlohmann::json::array()))
		fd_map[pair.value("vm_fd", 0)] =
			pair.value("host_map_id", 0);
	for (size_t i = 0; i < insns.size(); i++) {
		if (insns[i].code != LDDW_OPCODE)
			continue;
		if (insns[i].src_reg != BPF_PSEUDO_MAP_FD_LOCAL &&
		    insns[i].src_reg != BPF_PSEUDO_MAP_VALUE_LOCAL)
			continue;
		auto it = fd_map.find(insns[i].imm);
		if (it == fd_map.end())
			return error_response(EINVAL,
					      "prog_load: unresolved vm_fd " +
						      std::to_string(
							      insns[i].imm));
		insns[i].imm = it->second;
		i++; // LDDW occupies two instruction slots.
	}
	std::string name = req.value("name", "remote_prog");
	int prog_type = req.value("prog_type", 0);
	int id = bpftime_progs_create(-1, insns.data(), insns.size(),
				      name.c_str(), prog_type);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "prog_load failed");
	return nlohmann::json{ { "ok", true }, { "host_prog_id", id } };
}

nlohmann::json RemoteHost::handle_perf_event_open(const nlohmann::json &req)
{
	std::string kind = req.value("kind", "uprobe");
	pid_t pid = req.value("pid", -1);
	if (kind == "uprobe" || kind == "uretprobe") {
		std::string module = req.value("module", "");
		uint64_t offset = req.value("offset", 0ull);
		uint64_t ref_ctr_off = req.value("ref_ctr_off", 0ull);
		platform_translate_uprobe(module, offset);
		int id = bpftime_uprobe_create(-1, pid, module.c_str(),
					       offset, kind == "uretprobe",
					       (size_t)ref_ctr_off);
		if (id < 0)
			return error_response(errno ? errno : EINVAL,
					      "perf_event_open failed");
		return nlohmann::json{ { "ok", true },
				       { "host_perf_id", id } };
	}
	if (kind == "tracepoint") {
		int id = bpftime_tracepoint_create(-1, pid,
						   req.value("tp_id", 0));
		if (id < 0)
			return error_response(errno ? errno : EINVAL,
					      "tracepoint create failed");
		return nlohmann::json{ { "ok", true },
				       { "host_perf_id", id } };
	}
	// software / profile: backed by a software perf event handler.
	int cpu = req.value("cpu", 0);
	int64_t config = (int64_t)req.value("sample_period",
					     req.value("ref_ctr_off", 0ull));
	int id = bpftime_add_software_perf_event(cpu, 0, config);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "software event create failed");
	(void)bpftime_get_software_perf_event_raw_buffer(
		id, (1 + 64) * static_cast<size_t>(getpagesize()));
	return nlohmann::json{ { "ok", true }, { "host_perf_id", id } };
}

nlohmann::json RemoteHost::handle_bpf_link_create(const nlohmann::json &req)
{
	int perf_id = req.value("host_perf_id", -1);
	int prog_id = req.value("host_prog_id", -1);
	uint64_t cookie = req.value("cookie", 0ull);
	int id = bpftime_attach_perf_to_bpf_with_cookie(perf_id, prog_id,
							cookie);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "bpf_link_create failed");
	return nlohmann::json{ { "ok", true }, { "host_link_id", id } };
}

nlohmann::json RemoteHost::handle_perf_state(const nlohmann::json &req,
					     bool enable)
{
	int perf_id = req.value("host_perf_id", -1);
	int rc = enable ? bpftime_perf_event_enable(perf_id) :
			 bpftime_perf_event_disable(perf_id);
	if (rc != 0)
		return error_response(errno ? errno : EINVAL,
				      "perf_event_enable/disable failed");
	pid_t pid = req.value("pid", -1);
	if (enable && pid > 0)
		platform_refresh_agent(pid);
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json
RemoteHost::dispatch(const nlohmann::json &req,
		     const std::vector<uint8_t> &blob,
		     std::vector<uint8_t> &resp_blob)
{
	std::string op = req.value("op", "");
	try {
		if (op == "hello")
			return handle_hello();
		if (op == "proc_check")
			return handle_proc_check(req);
		if (op == "proc_maps")
			return handle_proc_maps(req);
		if (op == "fetch_elf" || op == "synth_elf")
			return handle_fetch_elf(req, resp_blob);
		if (op == "spawn_cmd")
			return handle_spawn(req);
		if (op == "resume_cmd")
			return handle_resume(req);

		std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
		if (op == "map_create")
			return handle_map_create(req);
		if (op == "map_lookup")
			return handle_map_lookup(req, blob, resp_blob);
		if (op == "map_update")
			return handle_map_update(req, blob);
		if (op == "map_delete")
			return handle_map_delete(req, blob);
		if (op == "map_get_next_key")
			return handle_map_get_next_key(req, blob, resp_blob);
		if (op == "prog_load")
			return handle_prog_load(req, blob);
		if (op == "perf_event_open")
			return handle_perf_event_open(req);
		if (op == "bpf_link_create" || op == "perf_event_attach_bpf")
			return handle_bpf_link_create(req);
		if (op == "perf_event_enable")
			return handle_perf_state(req, true);
		if (op == "perf_event_disable")
			return handle_perf_state(req, false);
		if (op == "close_obj") {
			int id = req.value("host_id", -1);
			if (id >= 0)
				bpftime_close(id);
			return nlohmann::json{ { "ok", true } };
		}
		return error_response(EINVAL, "unknown op: " + op);
	} catch (const std::exception &ex) {
		SPDLOG_ERROR("op {} threw: {}", op, ex.what());
		return error_response(EIO, ex.what());
	}
}

void RemoteHost::cleanup_session(const SessionResources &session)
{
	if (session.links.empty() && session.perfs.empty() &&
	    session.progs.empty() && session.maps.empty())
		return;
	{
		std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
		for (int id : session.links)
			bpftime_close(id);
		for (int id : session.perfs)
			bpftime_close(id);
		for (int id : session.progs)
			bpftime_close(id);
		for (int id : session.maps)
			bpftime_close(id);
	}
	for (pid_t pid : session.attached_pids) {
		if (pid > 0)
			platform_refresh_agent(pid);
	}
}

void RemoteHost::run_event_stream(int client_fd, std::atomic<bool> &stop)
{
	send_frame(client_fd, nlohmann::json{ { "ok", true } }.dump());
	auto &shm = shm_holder.global_shared_memory;
	const handler_manager *mgr = shm.get_manager();
	std::vector<QueuedEvent> batch;
	batch.reserve(128);
	while (!stop.load()) {
		// Detect half-open peers without consuming a frame.
		char probe = 0;
		ssize_t peek = recv(client_fd, &probe, 1,
				    MSG_PEEK | MSG_DONTWAIT);
		if (peek == 0 ||
		    (peek < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
		     errno != EINTR))
			break;
		batch.clear();
		{
			std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
			for (int fd = 0; mgr && fd < (int)mgr->size(); fd++) {
				if (stop.load() || !mgr->is_allocated(fd))
					continue;
				try {
					if (shm.is_ringbuf_map_fd(fd)) {
						RingCtx ctx{ fd, &stop,
							     &batch, 0 };
						(void)bpftime_poll_from_ringbuf(
							fd, &ctx,
							ringbuf_forward_cb);
					} else if (shm.is_perf_event_handler_fd(
							   fd)) {
						auto &handler = std::get<
							bpf_perf_event_handler>(
							mgr->get_handler(fd));
						if (!std::holds_alternative<
							    software_perf_event_shared_ptr>(
							    handler.data))
							continue;
						auto perf = std::get<
							software_perf_event_shared_ptr>(
							handler.data);
						std::vector<uint8_t> record;
						int drained = 0;
						while (drained <
							       kMaxEventsPerPollSlice &&
						       perf->drain_next_to_heap(
							       record)) {
							drained++;
							constexpr size_t kOff =
								sizeof(perf_event_header) +
								sizeof(uint32_t);
							if (record.size() < kOff)
								continue;
							uint32_t payload_size;
							memcpy(&payload_size,
							       record.data() +
								       sizeof(perf_event_header),
							       sizeof(uint32_t));
							size_t n = std::min<
								size_t>(
								payload_size,
								record.size() -
									kOff);
							batch.push_back(
								QueuedEvent{
									true, fd,
									std::vector<uint8_t>(
										record.data() +
											kOff,
										record.data() +
											kOff + n) });
						}
					}
				} catch (const std::exception &ex) {
					SPDLOG_DEBUG("event scan fd {}: {}",
						     fd, ex.what());
				}
			}
		}
		bool send_ok = true;
		for (const auto &ev : batch) {
			if (stop.load())
				break;
			nlohmann::json m{
				{ "op", "event" },
				{ "map_id", ev.fd },
				{ "type", ev.is_perf ? "perf_event" :
						       "ringbuf" }
			};
			try {
				send_frame(client_fd, m.dump(),
					   ev.payload.data(), ev.payload.size());
			} catch (...) {
				send_ok = false;
				break;
			}
		}
		if (!send_ok)
			break;
		if (batch.empty())
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1));
	}
}

} // namespace remote
} // namespace bpftime
