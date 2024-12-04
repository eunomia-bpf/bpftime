#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
#include <ostream>
#include <string>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <variant>
#include <vector>
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpf_attach_ctx.hpp"
#include <bpftime_shm_internal.hpp>
#include <fstream>
#include <time.h>
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;
using namespace bpftime;

static int run_ebpf_and_measure(bpftime_prog& prog, std::vector<uint8_t>& data_in, int repeat_N) {
    // Test the program
    uint64_t return_val;
    void* memory = data_in.data();
    size_t memory_size = data_in.size();

    // Start timer
    auto start = std::chrono::steady_clock::now();

    // Run the program
    for (int i = 0; i < repeat_N; i++) {
        prog.bpftime_prog_exec(memory, memory_size, &return_val);
    }

    // End timer
    auto end = std::chrono::steady_clock::now();

    // Calculate average time in ns
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / repeat_N;

    std::cout << "Time taken: " << time_ns << " ns" << std::endl;
    std::cout << "Return value: " << return_val << std::endl;

    return 0;
}

int bpftime_run_ebpf_program(int id,
			     const std::string &data_in_file, int repeat_N, const std::string &run_type)
{
	cout << "Running eBPF program with id " << id << " and data in file " << data_in_file << endl;
	cout << "Repeat N: " << repeat_N << " with run type " << run_type << endl;
	std::vector<uint8_t> data_in;
	std::ifstream ifs(data_in_file, std::ios::binary | std::ios::ate);
	if (!ifs.is_open()) {
		cerr << "Unable to open data in file" << endl;
		return 1;
	}
	std::streamsize size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	data_in.resize(size);
	if (!ifs.read((char *)data_in.data(), size)) {
		cerr << "Unable to read data in file" << endl;
		return 1;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	size_t handler_size = manager->size();
	if ((size_t) id >= handler_size || id < 0) {
		cerr << "Invalid id " << id << " not exist" << endl;
		return 1;
	}
	if (std::holds_alternative<bpf_prog_handler>(
			    manager->get_handler(id))) {
			const auto &prog = std::get<bpf_prog_handler>(
				manager->get_handler(id));
			auto new_prog = bpftime_prog(prog.insns.data(),
							 prog.insns.size(),
							 prog.name.c_str());
					bpftime::bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&new_prog);
			bpftime::bpftime_helper_group::get_shm_maps_helper_group()
				.add_helper_group_to_prog(&new_prog);
			if (run_type == "JIT") {
				new_prog.bpftime_prog_load(true);
			} else if (run_type == "AOT") {
				if (prog.aot_insns.size() == 0) {
					cerr << "AOT instructions not found" << endl;
					return 1;
				}
				new_prog.load_aot_object(std::vector<uint8_t>(prog.aot_insns.begin(), prog.aot_insns.end()));
			} else if (run_type == "INTERPRET") {
				new_prog.bpftime_prog_load(false);
			}
			return run_ebpf_and_measure(new_prog, data_in, repeat_N);
	} else {
		cerr << "Invalid id " << id << " not a bpf program" << endl;
		return 1;
	}
	return 0;
}

inline void check_run_type(const std::string &run_type) {
	if (run_type != "JIT" && run_type != "AOT" && run_type != "INTERPRET") {
		cerr << "Invalid run type " << run_type << endl;
		cerr << "Valid run types are JIT, AOT, INTERPRET" << endl;
		exit(1);
	}
}

// Function pointer types for original libc functions
typedef int (*orig_fcntl_t)(int fd, int cmd, ...);
typedef int (*orig_open_t)(const char *pathname, int flags, ...);
typedef int (*orig_openat_t)(int dirfd, const char *pathname, int flags, ...);
typedef int (*orig_close_t)(int fd);

// Store original function pointers
static orig_fcntl_t orig_fcntl = nullptr;
static orig_open_t orig_open = nullptr;
static orig_openat_t orig_openat = nullptr;
static orig_close_t orig_close = nullptr;

// Global pointers to eBPF programs for better performance
static bpftime_prog* fcntl_prog = nullptr;
static bpftime_prog* open_prog = nullptr;
static bpftime_prog* openat_prog = nullptr;
static bpftime_prog* close_prog = nullptr;

// Initialize eBPF environment and load programs
static void __attribute__((constructor)) init_library() {
    // Initialize shared memory
    bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
    const handler_manager* manager = shm_holder.global_shared_memory.get_manager();

    // Load original libc functions
    orig_fcntl = (orig_fcntl_t)dlsym(RTLD_NEXT, "fcntl");
    orig_open = (orig_open_t)dlsym(RTLD_NEXT, "open");
    orig_openat = (orig_openat_t)dlsym(RTLD_NEXT, "openat");
    orig_close = (orig_close_t)dlsym(RTLD_NEXT, "close");

    // Load all eBPF programs
    for (size_t i = 0; i < manager->size(); i++) {
        if (std::holds_alternative<bpf_prog_handler>(manager->get_handler(i))) {
            const auto& prog = std::get<bpf_prog_handler>(manager->get_handler(i));
            
            auto load_prog = [](const bpf_prog_handler& prog) -> bpftime_prog* {
                auto new_prog = new bpftime_prog(prog.insns.data(), prog.insns.size(), prog.name.c_str());
                bpftime::bpftime_helper_group::get_kernel_utils_helper_group().add_helper_group_to_prog(new_prog);
                bpftime::bpftime_helper_group::get_shm_maps_helper_group().add_helper_group_to_prog(new_prog);
                new_prog->bpftime_prog_load(true);  // Use JIT by default
                return new_prog;
            };

            if (prog.name == "uprobe__libc_fcntl") {
                fcntl_prog = load_prog(prog);
            } else if (prog.name == "uprobe__libc_open") {
                open_prog = load_prog(prog);
            } else if (prog.name == "uprobe__libc_openat") {
                openat_prog = load_prog(prog);
            } else if (prog.name == "uprobe__libc_close") {
                close_prog = load_prog(prog);
            }
        }
    }
}

// Cleanup function
static void __attribute__((destructor)) cleanup_library() {
    delete fcntl_prog;
    delete open_prog;
    delete openat_prog;
    delete close_prog;
}

// Hook implementations
extern "C" {

int fcntl(int fd, int cmd, ...) {
    if (fcntl_prog) {
        uint64_t ret;
        struct {
            int fd;
            int cmd;
        } ctx = {fd, cmd};
        fcntl_prog->bpftime_prog_exec(&ctx, sizeof(ctx), &ret);
    }
    va_list args;
    va_start(args, cmd);
    void* arg = va_arg(args, void*);
    int result = orig_fcntl(fd, cmd, arg);
    va_end(args);
    return result;
}

int open(const char *pathname, int flags, ...) {
    if (open_prog) {
        uint64_t ret;
        struct {
            const char* pathname;
            int flags;
        } ctx = {pathname, flags};
        open_prog->bpftime_prog_exec(&ctx, sizeof(ctx), &ret);
    }
    va_list args;
    va_start(args, flags);
    mode_t mode = va_arg(args, mode_t);
    int result = orig_open(pathname, flags, mode);
    va_end(args);
    return result;
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    if (openat_prog) {
        uint64_t ret;
        struct {
            int dirfd;
            const char* pathname;
            int flags;
        } ctx = {dirfd, pathname, flags};
        openat_prog->bpftime_prog_exec(&ctx, sizeof(ctx), &ret);
    }
    va_list args;
    va_start(args, flags);
    mode_t mode = va_arg(args, mode_t);
    int result = orig_openat(dirfd, pathname, flags, mode);
    va_end(args);
    return result;
}

int close(int fd) {
    if (close_prog) {
        uint64_t ret;
        struct {
            int fd;
        } ctx = {fd};
        close_prog->bpftime_prog_exec(&ctx, sizeof(ctx), &ret);
    }
    return orig_close(fd);
}

}
