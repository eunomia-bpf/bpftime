/* SPDX-License-Identifier: MIT
 *
 * bpftime-darwin-host implementation.
 */
#include "darwin_host.hpp"
#include "elf_synth.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "ebpf-vm.h"
#include "handler/perf_event_handler.hpp"

#include <fcntl.h>
#include <libproc.h>
#include <signal.h>
#include <spawn.h>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
extern char **environ;
#include <fstream>
#include <set>
#include <thread>

namespace bpftime
{
namespace darwin
{

namespace
{
constexpr int BPF_PSEUDO_MAP_FD_LOCAL = 1;
constexpr int BPF_PSEUDO_MAP_VALUE_LOCAL = 2;
constexpr uint8_t LDDW_OPCODE = 0x18;

uint32_t align8(uint32_t v)
{
	return (v + 7u) & ~7u;
}

bool is_percpu_type(int type)
{
	return type == 5 || type == 6; // BPF_MAP_TYPE_PERCPU_HASH / ARRAY
}

std::string tmpdir()
{
	const char *t = getenv("TMPDIR");
	return (t && t[0]) ? std::string(t) : std::string("/tmp/");
}

// Rewrite dyld_shared_cache image paths to their /bpftime-dyld virtual form
// inside a synthesized /proc/<pid>/maps document.
std::string virtualize_maps(const std::string &text)
{
	const std::string prefix = "/bpftime-dyld";
	std::string out;
	out.reserve(text.size() + 64);
	size_t pos = 0;
	while (pos < text.size()) {
		size_t eol = text.find('\n', pos);
		if (eol == std::string::npos)
			eol = text.size();
		std::string line = text.substr(pos, eol - pos);
		for (const char *pref : { "/usr/lib/", "/System/Library/" }) {
			size_t p = line.find(pref);
			if (p != std::string::npos) {
				line.insert(p, prefix);
				break;
			}
		}
		out += line;
		out.push_back('\n');
		pos = eol + 1;
	}
	return out;
}
} // namespace

nlohmann::json error_response(int err, const std::string &msg)
{
	return nlohmann::json{ { "ok", false },
			       { "err", err },
			       { "msg", msg } };
}

DarwinHost::DarwinHost(std::string agent_dylib_path)
	: agent_dylib_path_(std::move(agent_dylib_path)),
	  shm_name_("bpftime_darwin_shm_" + std::to_string((uint64_t)getpid()))
{
}

DarwinHost::~DarwinHost()
{
	shutdown_shm();
}

void DarwinHost::initialize_shm()
{
	setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name_.c_str(), 1);
	// BCC creates a 256 MiB RINGBUF that bpftime double-backs; keep a
	// 1 GiB default so sessions never hit bad_alloc. Overridable via env.
	if (getenv("BPFTIME_SHM_MEMORY_MB") == nullptr)
		setenv("BPFTIME_SHM_MEMORY_MB", "1024", 1);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	SPDLOG_INFO("darwin host: created SHM segment {}", shm_name_);
}

void DarwinHost::shutdown_shm()
{
	if (bpftime_global_shm_is_initialized()) {
		bpftime_destroy_global_shm();
		bpftime_remove_global_shm();
	}
}

const ModuleSymbols &DarwinHost::resolve_module_symbols(const std::string &path)
{
	std::lock_guard<std::mutex> guard(symbol_mutex_);
	auto cached = symbol_cache_.find(path);
	if (cached != symbol_cache_.end())
		return cached->second;
	ModuleSymbols mod;
	if (is_dyld_virtual_path(path)) {
		std::string real = dyld_virtual_to_real(path);
		if (real == "/usr/lib/libSystem.B.dylib") {
			auto [it, inserted] = umbrella_cache_.emplace(
				path, build_libsystem_umbrella(path));
			mod = it->second.stub;
		} else {
			mod = enumerate_runtime_exports(real);
			mod.canonical_path = path;
		}
	} else {
		mod = parse_disk_macho(path);
	}
	auto [it, _] = symbol_cache_.emplace(path, std::move(mod));
	return it->second;
}

nlohmann::json DarwinHost::handle_hello()
{
	return nlohmann::json{ { "ok", true },
			       { "arch", "arm64" },
			       { "num_cpus",
				 (int)sysconf(_SC_NPROCESSORS_ONLN) },
			       { "host_pid", (int)getpid() },
			       { "shm", shm_name_ } };
}

nlohmann::json DarwinHost::handle_proc_check(const nlohmann::json &req)
{
	pid_t pid = req.value("pid", 0);
	if (pid <= 0 || kill(pid, 0) != 0)
		return nlohmann::json{ { "ok", true },
				       { "alive", false },
				       { "comm", "" },
				       { "exe", "" } };
	char path[PROC_PIDPATHINFO_MAXSIZE] = {};
	int n = proc_pidpath(pid, path, sizeof(path));
	std::string exe = n > 0 ? std::string(path) : std::string();
	std::string comm;
	if (!exe.empty()) {
		size_t slash = exe.find_last_of('/');
		comm = slash == std::string::npos ? exe : exe.substr(slash + 1);
	}
	return nlohmann::json{ { "ok", true },
			       { "alive", true },
			       { "comm", comm },
			       { "exe", exe } };
}

nlohmann::json DarwinHost::handle_proc_maps(const nlohmann::json &req,
					    std::vector<uint8_t> &resp_blob)
{
	pid_t pid = req.value("pid", 0);
	std::string maps_path =
		"/tmp/bpftime-maps-" + std::to_string(pid) + ".txt";
	std::ifstream f(maps_path);
	if (!f)
		return error_response(ENOENT, "maps file not found for pid " +
						      std::to_string(pid));
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	content = virtualize_maps(content);
	resp_blob.assign(content.begin(), content.end());
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json DarwinHost::handle_resolve_lib(const nlohmann::json &req)
{
	std::string name = req.value("name", "");
	static const std::unordered_map<std::string, std::string> kAliases = {
		{ "c", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
		{ "libc", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
		{ "libc.so.6", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
		{ "pthread", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
		{ "m", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
		{ "dl", "/bpftime-dyld/usr/lib/libSystem.B.dylib" },
	};
	auto it = kAliases.find(name);
	std::string path = it != kAliases.end() ?
				   it->second :
				   "/bpftime-dyld/usr/lib/" + name + ".dylib";
	return nlohmann::json{ { "ok", true }, { "path", path } };
}

nlohmann::json DarwinHost::handle_synth_elf(const nlohmann::json &req,
					    std::vector<uint8_t> &resp_blob)
{
	std::string path = req.value("path", "");
	if (path.empty())
		return error_response(EINVAL, "missing path");
	const ModuleSymbols &mod = resolve_module_symbols(path);
	if (mod.symbols.empty())
		SPDLOG_WARN("synth_elf: no symbols extracted for {}", path);
	resp_blob = build_stub_elf(mod);
	return nlohmann::json{ { "ok", true },
			       { "canonical_path", mod.canonical_path } };
}

nlohmann::json DarwinHost::handle_map_create(const nlohmann::json &req)
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

nlohmann::json DarwinHost::handle_map_lookup(const nlohmann::json &req,
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
	return nlohmann::json{ { "ok", true },
			       { "is_percpu", false },
			       { "num_cpus", 1 } };
}

nlohmann::json DarwinHost::handle_map_update(const nlohmann::json &req,
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
		need = attr.key_size + (size_t)sysconf(_SC_NPROCESSORS_ONLN) *
					       align8(attr.value_size);
	if (blob.size() < need)
		return error_response(EINVAL, "blob too short");
	const void *val_ptr = blob.data() + attr.key_size;
	int dup_inner = -1;
	if ((attr.type == 12 || attr.type == 13) &&
	    attr.value_size >= sizeof(int32_t)) {
		int inner_id = *reinterpret_cast<const int32_t *>(val_ptr);
		int owned = bpftime_maps_dup(inner_id, -1);
		if (owned >= 0) {
			dup_inner = owned;
			val_ptr = &dup_inner;
		}
	}
	long rc = bpftime_map_update_elem(id, blob.data(), val_ptr, flags);
	if (rc != 0)
		return error_response(errno ? errno : EIO, "update failed");
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json
DarwinHost::handle_map_update_batch(const nlohmann::json &req,
				    const std::vector<uint8_t> &blob)
{
	int id = req.value("host_map_id", -1);
	uint64_t flags = req.value("flags", 0ull);
	uint32_t count = req.value("count", 0u);
	bpftime::bpf_map_attr attr;
	const char *name = nullptr;
	bpftime::bpf_map_type type;
	if (bpftime_map_get_info(id, &attr, &name, &type) != 0)
		return error_response(ENOENT, "unknown map id");
	size_t vsz = attr.value_size;
	if (is_percpu_type(attr.type))
		vsz = (size_t)sysconf(_SC_NPROCESSORS_ONLN) *
		      align8(attr.value_size);
	size_t stride = attr.key_size + vsz;
	if (blob.size() < size_t(count) * stride)
		return error_response(EINVAL, "batch blob too short");
	uint32_t updated = 0;
	for (; updated < count; updated++) {
		const uint8_t *entry = blob.data() + size_t(updated) * stride;
		const void *val_ptr = entry + attr.key_size;
		int dup_inner = -1;
		if ((attr.type == 12 || attr.type == 13) &&
		    attr.value_size >= sizeof(int32_t)) {
			int inner_id =
				*reinterpret_cast<const int32_t *>(val_ptr);
			int owned = bpftime_maps_dup(inner_id, -1);
			if (owned >= 0) {
				dup_inner = owned;
				val_ptr = &dup_inner;
			}
		}
		if (bpftime_map_update_elem(id, entry, val_ptr, flags) != 0)
			return error_response(errno ? errno : EIO,
					      "batch update failed");
	}
	return nlohmann::json{ { "ok", true }, { "updated", updated } };
}

nlohmann::json DarwinHost::handle_map_delete(const nlohmann::json &req,
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
	long rc = bpftime_map_delete_elem(id, blob.data());
	if (rc != 0)
		return error_response(errno ? errno : EIO, "delete failed");
	return nlohmann::json{ { "ok", true } };
}

nlohmann::json
DarwinHost::handle_map_get_next_key(const nlohmann::json &req,
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
			return nlohmann::json{ { "ok", true },
					       { "end", true } };
		return error_response(errno ? errno : EIO,
				      "get_next_key failed");
	}
	resp_blob = std::move(next);
	return nlohmann::json{ { "ok", true }, { "end", false } };
}

nlohmann::json DarwinHost::handle_prog_load(const nlohmann::json &req,
					    const std::vector<uint8_t> &blob)
{
	if (blob.size() % sizeof(ebpf_inst) != 0)
		return error_response(EINVAL, "insn blob not 8-byte aligned");
	std::vector<ebpf_inst> insns(blob.size() / sizeof(ebpf_inst));
	memcpy(insns.data(), blob.data(), blob.size());

	std::unordered_map<int, int> fd_map;
	for (const auto &pair : req.value("fd_map", nlohmann::json::array())) {
		fd_map[pair.value("vm_fd", 0)] = pair.value("host_map_id", 0);
	}
	for (size_t i = 0; i < insns.size(); i++) {
		if (insns[i].code != LDDW_OPCODE)
			continue;
		if (insns[i].src_reg != BPF_PSEUDO_MAP_FD_LOCAL &&
		    insns[i].src_reg != BPF_PSEUDO_MAP_VALUE_LOCAL)
			continue;
		auto it = fd_map.find(insns[i].imm);
		if (it == fd_map.end())
			return error_response(
				EINVAL, "prog_load: unresolved vm_fd " +
						std::to_string(insns[i].imm));
		insns[i].imm = it->second;
		i++; // LDDW occupies two instruction slots.
	}
	std::string name = req.value("name", "darwin_prog");
	int prog_type = req.value("prog_type", 0);
	int id = bpftime_progs_create(-1, insns.data(), insns.size(),
				      name.c_str(), prog_type);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "prog_load failed");
	return nlohmann::json{ { "ok", true }, { "host_prog_id", id } };
}

nlohmann::json DarwinHost::handle_perf_event_open(const nlohmann::json &req)
{
	std::string kind = req.value("kind", "uprobe");
	pid_t pid = req.value("pid", -1);
	if (kind == "uprobe" || kind == "uretprobe") {
		std::string module = req.value("module", "");
		uint64_t offset = req.value("offset", 0ull);
		uint64_t ref_ctr_off = req.value("ref_ctr_off", 0ull);
		// Translate libSystem umbrella synthetic offsets.
		if (module == "/bpftime-dyld/usr/lib/libSystem.B.dylib" ||
		    module == "/usr/lib/libSystem.B.dylib") {
			auto it = umbrella_cache_.find(
				"/bpftime-dyld/usr/lib/libSystem.B.dylib");
			if (it != umbrella_cache_.end()) {
				std::string sym_name;
				for (const auto &s : it->second.stub.symbols) {
					if (s.offset == offset) {
						sym_name = s.name;
						break;
					}
				}
				auto dit =
					it->second.deep_symbols.find(sym_name);
				if (dit != it->second.deep_symbols.end()) {
					module = dit->second.module;
					offset = dit->second.offset;
				} else {
					auto tit = it->second.translations.find(
						offset);
					if (tit !=
					    it->second.translations.end()) {
						module = tit->second.first;
						offset = tit->second.second;
					}
				}
			}
		} else if (is_dyld_virtual_path(module)) {
			module = dyld_virtual_to_real(module);
		}
		bool retprobe = (kind == "uretprobe");
		int id = bpftime_uprobe_create(-1, pid, module.c_str(), offset,
					       retprobe, (size_t)ref_ctr_off);
		if (id < 0)
			return error_response(errno ? errno : EINVAL,
					      "perf_event_open failed");
		return nlohmann::json{ { "ok", true }, { "host_perf_id", id } };
	}
	if (kind == "tracepoint") {
		int id = bpftime_tracepoint_create(-1, pid,
						   req.value("tp_id", 0));
		if (id < 0)
			return error_response(errno ? errno : EINVAL,
					      "tracepoint create failed");
		return nlohmann::json{ { "ok", true }, { "host_perf_id", id } };
	}
	// software / profile: back with a software perf event handler.
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

nlohmann::json DarwinHost::handle_perf_attach(const nlohmann::json &req)
{
	int perf_id = req.value("host_perf_id", -1);
	int prog_id = req.value("host_prog_id", -1);
	uint64_t cookie = req.value("cookie", 0ull);
	int id = bpftime_attach_perf_to_bpf_with_cookie(perf_id, prog_id,
							cookie);
	if (id < 0)
		return error_response(errno ? errno : EINVAL,
				      "perf_event_attach_bpf failed");
	return nlohmann::json{ { "ok", true }, { "host_link_id", id } };
}

nlohmann::json DarwinHost::handle_perf_enable(const nlohmann::json &req)
{
	int perf_id = req.value("host_perf_id", -1);
	bool enable = req.value("enable", true);
	int rc = enable ? bpftime_perf_event_enable(perf_id) :
			  bpftime_perf_event_disable(perf_id);
	if (rc != 0)
		return error_response(errno ? errno : EINVAL,
				      "perf_event_enable failed");
	pid_t pid = req.value("pid", -1);
	if (enable && pid > 0)
		refresh_agent(pid);
	return nlohmann::json{ { "ok", true } };
}

void DarwinHost::refresh_agent(pid_t pid)
{
	std::string path =
		tmpdir() + "bpftime-agent-" + std::to_string(pid) + ".sock";
	struct stat st;
	if (stat(path.c_str(), &st) != 0) {
		SPDLOG_DEBUG("refresh: agent socket {} not present yet", path);
		return;
	}
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0)
		return;
	sockaddr_un addr{};
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
	if (connect(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
		close(fd);
		return;
	}
	const char msg[] = "refresh";
	(void)send(fd, msg, sizeof(msg) - 1, 0);
	char buf[64];
	(void)recv(fd, buf, sizeof(buf), 0);
	close(fd);
	SPDLOG_INFO("refresh: sent incremental attach to pid {}", (int)pid);
}

nlohmann::json DarwinHost::handle_spawn(const nlohmann::json &req)
{
	auto argv = req.value("argv", nlohmann::json::array());
	if (argv.empty() || !argv[0].is_string())
		return error_response(EINVAL, "argv required");
	bool hold = req.value("hold_at_entry", false);
	std::string cwd = req.value("cwd", "");

	std::vector<std::string> argv_storage;
	argv_storage.reserve(argv.size());
	for (const auto &a : argv)
		argv_storage.push_back(a.get<std::string>());
	std::vector<char *> cargv;
	cargv.reserve(argv_storage.size() + 1);
	for (const auto &a : argv_storage)
		cargv.push_back(const_cast<char *>(a.c_str()));
	cargv.push_back(nullptr);

	// Environment assembled in the parent so posix_spawn only touches
	// async-signal-safe internals. BPFTIME_ENABLE_AGENT_IPC lets the
	// Darwin preload agent accept incremental attach over its socket.
	std::vector<std::string> env_storage;
	// Skip injection when no readable agent dylib is installed; dyld
	// aborts processes named in DYLD_INSERT_LIBRARIES but missing.
	if (access(agent_dylib_path_.c_str(), R_OK) == 0)
		env_storage.emplace_back("DYLD_INSERT_LIBRARIES=" +
					 agent_dylib_path_);
	env_storage.emplace_back("BPFTIME_GLOBAL_SHM_NAME=" + shm_name_);
	env_storage.emplace_back("BPFTIME_ENABLE_AGENT_IPC=1");
	if (req.contains("env") && req["env"].is_object()) {
		for (auto it = req["env"].begin(); it != req["env"].end(); ++it)
			env_storage.emplace_back(it.key() + "=" +
						 it.value().get<std::string>());
	}
	static const char *const kOwnedKeys[] = {
		"DYLD_INSERT_LIBRARIES",
		"BPFTIME_GLOBAL_SHM_NAME",
		"BPFTIME_ENABLE_AGENT_IPC",
	};
	if (environ != nullptr) {
		for (char **e = environ; *e != nullptr; ++e) {
			std::string entry(*e);
			size_t eq = entry.find('=');
			std::string key = eq == std::string::npos ?
						  entry :
						  entry.substr(0, eq);
			bool owned = false;
			for (const char *k : kOwnedKeys) {
				if (key == k) {
					owned = true;
					break;
				}
			}
			if (owned)
				continue;
			if (req.contains("env") && req["env"].is_object() &&
			    req["env"].contains(key))
				continue;
			env_storage.push_back(std::move(entry));
		}
	}
	std::vector<char *> cenvp;
	cenvp.reserve(env_storage.size() + 1);
	for (const auto &e : env_storage)
		cenvp.push_back(const_cast<char *>(e.c_str()));
	cenvp.push_back(nullptr);

	int out_pipe[2] = { -1, -1 };
	int err_pipe[2] = { -1, -1 };
	if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0)
		return error_response(errno, "pipe failed");
	for (int p : { out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1] })
		fcntl(p, F_SETFD, FD_CLOEXEC);

	posix_spawn_file_actions_t actions;
	posix_spawnattr_t attr;
	if (posix_spawn_file_actions_init(&actions) != 0 ||
	    posix_spawnattr_init(&attr) != 0) {
		close(out_pipe[0]);
		close(out_pipe[1]);
		close(err_pipe[0]);
		close(err_pipe[1]);
		return error_response(errno, "spawn init failed");
	}
	// Cold-start hold: the kernel suspends the new Mach-O process before
	// dyld runs; resume_cmd releases it with SIGCONT. No FIFO needed.
	if (hold &&
	    posix_spawnattr_setflags(&attr, POSIX_SPAWN_START_SUSPENDED) != 0) {
		posix_spawn_file_actions_destroy(&actions);
		posix_spawnattr_destroy(&attr);
		return error_response(errno, "posix_spawnattr_setflags failed");
	}
	posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
	posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
	posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
	posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
	posix_spawn_file_actions_addclose(&actions, err_pipe[1]);
	if (!cwd.empty())
		posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());

	auto st = std::make_unique<SpawnState>();
	st->spawn_id = next_spawn_id_++;
	st->held = hold;
	pid_t pid = -1;
	int rc = posix_spawn(&pid, cargv[0], &actions, &attr, cargv.data(),
			     cenvp.data());
	posix_spawn_file_actions_destroy(&actions);
	posix_spawnattr_destroy(&attr);
	for (int p : { out_pipe[1], err_pipe[1] })
		close(p);
	if (rc != 0) {
		close(out_pipe[0]);
		close(err_pipe[0]);
		return error_response(rc, "posix_spawn failed");
	}

	st->pid = pid;
	st->out_read_fd = out_pipe[0];
	st->err_read_fd = err_pipe[0];
	SpawnState *raw = st.get();
	auto drain = [raw](int fd) {
		uint8_t buf[4096];
		for (;;) {
			ssize_t n = read(fd, buf, sizeof(buf));
			if (n <= 0)
				return;
			std::lock_guard<std::mutex> g(raw->output_mutex);
			raw->captured_output.insert(raw->captured_output.end(),
						    buf, buf + n);
		}
	};
	std::thread(drain, out_pipe[0]).detach();
	std::thread(drain, err_pipe[0]).detach();

	int spawn_id = st->spawn_id;
	{
		std::lock_guard<std::mutex> g(spawn_mutex_);
		spawns_[spawn_id] = std::move(st);
	}
	return nlohmann::json{ { "ok", true },
			       { "spawn_id", spawn_id },
			       { "host_pid", (int)pid } };
}

void DarwinHost::reap_spawn(SpawnState &st)
{
	if (st.exited.load() || st.pid < 0)
		return;
	int status = 0;
	pid_t r = waitpid(st.pid, &status, WNOHANG);
	if (r != st.pid)
		return;
	if (WIFEXITED(status)) {
		st.exit_code = WEXITSTATUS(status);
	} else if (WIFSIGNALED(status)) {
		st.term_sig = WTERMSIG(status);
		st.exit_code = -1;
	}
	st.exited = true;
	if (st.out_read_fd >= 0) {
		close(st.out_read_fd);
		close(st.err_read_fd);
		st.out_read_fd = st.err_read_fd = -1;
	}
}

nlohmann::json DarwinHost::handle_cmd_ctl(const nlohmann::json &req,
					  std::vector<uint8_t> &resp_blob)
{
	int spawn_id = req.value("spawn_id", -1);
	std::string action = req.value("action", "wait_poll");
	SpawnState *st = nullptr;
	{
		std::lock_guard<std::mutex> g(spawn_mutex_);
		auto it = spawns_.find(spawn_id);
		if (it == spawns_.end())
			return error_response(ENOENT, "unknown spawn_id");
		st = it->second.get();
	}
	if (action == "resume" && st->held) {
		// Release the POSIX_SPAWN_START_SUSPENDED hold.
		kill(st->pid, SIGCONT);
		st->held = false;
	} else if (action == "kill") {
		int sig = req.value("sig", SIGTERM);
		if (sig <= 0)
			sig = SIGTERM;
		// A suspended process only acts on the signal after resume.
		if (st->held) {
			kill(st->pid, SIGCONT);
			st->held = false;
		}
		kill(st->pid, sig);
	}
	reap_spawn(*st);
	{
		std::lock_guard<std::mutex> g(st->output_mutex);
		resp_blob = st->captured_output;
		st->captured_output.clear();
	}
	return nlohmann::json{ { "ok", true },
			       { "exited", st->exited.load() },
			       { "exit_code", st->exit_code.load() },
			       { "term_sig", st->term_sig.load() } };
}

nlohmann::json DarwinHost::dispatch(const nlohmann::json &req,
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
			return handle_proc_maps(req, resp_blob);
		if (op == "resolve_lib")
			return handle_resolve_lib(req);
		if (op == "synth_elf")
			return handle_synth_elf(req, resp_blob);
		if (op == "spawn_cmd")
			return handle_spawn(req);
		if (op == "cmd_ctl")
			return handle_cmd_ctl(req, resp_blob);
		std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
		if (op == "map_create")
			return handle_map_create(req);
		if (op == "map_lookup")
			return handle_map_lookup(req, blob, resp_blob);
		if (op == "map_update")
			return handle_map_update(req, blob);
		if (op == "map_update_batch")
			return handle_map_update_batch(req, blob);
		if (op == "map_delete")
			return handle_map_delete(req, blob);
		if (op == "map_get_next_key")
			return handle_map_get_next_key(req, blob, resp_blob);
		if (op == "prog_load")
			return handle_prog_load(req, blob);
		if (op == "perf_event_open")
			return handle_perf_event_open(req);
		if (op == "perf_event_attach_bpf")
			return handle_perf_attach(req);
		if (op == "perf_event_enable")
			return handle_perf_enable(req);
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

void DarwinHost::cleanup_session(const SessionResources &session)
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
	}
	std::set<pid_t> unique_pids(session.attached_pids.begin(),
				    session.attached_pids.end());
	for (pid_t pid : unique_pids) {
		if (pid > 0 && kill(pid, 0) == 0)
			refresh_agent(pid);
	}
	{
		std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
		for (int id : session.progs)
			bpftime_close(id);
		for (int id : session.maps)
			bpftime_close(id);
	}
}

namespace
{
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

constexpr int kMaxEventsPerPollSlice = 64;

int ringbuf_forward_cb(void *raw, void *data, size_t len)
{
	auto *c = static_cast<RingCtx *>(raw);
	if (c->stop->load())
		return -1;
	QueuedEvent ev;
	ev.is_perf = false;
	ev.fd = c->fd;
	const auto *bytes = static_cast<const uint8_t *>(data);
	ev.payload.assign(bytes, bytes + len);
	c->out->push_back(std::move(ev));
	// Bound lock hold time by committing the consumer position at 64.
	if (++c->count >= kMaxEventsPerPollSlice)
		return -1;
	return 0;
}
} // namespace

void DarwinHost::run_event_stream(int client_fd, std::atomic<bool> &stop)
{
	using namespace darwin_bridge;
	send_frame(client_fd, nlohmann::json{ { "ok", true } }.dump());
	auto &shm = shm_holder.global_shared_memory;
	const handler_manager *mgr = shm.get_manager();
	std::vector<QueuedEvent> batch;
	batch.reserve(128);
	while (!stop.load()) {
		char eof_probe = 0;
		ssize_t peek =
			recv(client_fd, &eof_probe, 1, MSG_PEEK | MSG_DONTWAIT);
		if (peek == 0 || (peek < 0 && errno != EAGAIN &&
				  errno != EWOULDBLOCK && errno != EINTR))
			break;
		batch.clear();
		{
			std::lock_guard<std::mutex> shm_guard(shm_op_mutex_);
			for (int fd = 0; mgr && fd < (int)mgr->size(); fd++) {
				if (stop.load() || !mgr->is_allocated(fd))
					continue;
				try {
					if (shm.is_ringbuf_map_fd(fd)) {
						RingCtx ctx{ fd, &stop, &batch,
							     0 };
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
							constexpr size_t kPayloadOff =
								sizeof(perf_event_header) +
								sizeof(uint32_t);
							if (record.size() <
							    kPayloadOff)
								continue;
							uint32_t payload_size;
							memcpy(&payload_size,
							       record.data() +
								       sizeof(perf_event_header),
							       sizeof(uint32_t));
							size_t avail =
								record.size() -
								kPayloadOff;
							size_t n = std::min<
								size_t>(
								payload_size,
								avail);
							QueuedEvent ev;
							ev.is_perf = true;
							ev.fd = fd;
							ev.payload.assign(
								record.data() +
									kPayloadOff,
								record.data() +
									kPayloadOff +
									n);
							batch.push_back(
								std::move(ev));
						}
					}
				} catch (const std::exception &ex) {
					SPDLOG_DEBUG(
						"event stream scan fd {}: {}",
						fd, ex.what());
				}
			}
		}
		bool send_ok = true;
		for (const auto &ev : batch) {
			if (stop.load())
				break;
			nlohmann::json m =
				ev.is_perf ?
					nlohmann::json{ { "op", "perf_event" },
							{ "host_map_id",
							  ev.fd },
							{ "cpu", 0 } } :
					nlohmann::json{
						{ "op", "ringbuf_event" },
						{ "host_map_id", ev.fd }
					};
			try {
				send_frame(client_fd, m.dump(),
					   ev.payload.data(),
					   ev.payload.size());
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

} // namespace darwin
} // namespace bpftime
