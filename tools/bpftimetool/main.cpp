#include <boost/program_options/value_semantic.hpp>
#include <cerrno>
#include <cstdlib>
#include <frida-core.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
#include <ostream>
#include <string>
#include <filesystem>
extern "C" {
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sys/stat.h>
}

const char *DEFAULT_INSTALLATION_LOCATION = "~/.bpftime";

using namespace std;

const std::string version = "1.0.0";
const bpftime::shm_open_type bpftime::global_shm_open_type =
	shm_open_type::SHM_NO_CREATE;

struct cli_config {
	bool dry_run = false;
	bool benchmark = false;
	bool kernel_uprobe = false;
	int pid = 0;
	std::string input_file;
	int target_pid;
	std::string agent_dynlib_path;
	std::string bpf_object_path;
	bool inject_client = false;
	bool inject_server = false;
};

using libbpf_object_ptr =
	std::unique_ptr<struct bpf_object, decltype(&bpf_object__close)>;

// open, load and attach the
static libbpf_object_ptr open_and_attach_libbpf_object(const char *filename)
{
	struct bpf_object *obj;
	struct bpf_program *prog;
	// struct bpf_link *link;
	int err;

	obj = bpf_object__open(filename);
	if (libbpf_get_error(obj)) {
		fprintf(stderr, "Error opening BPF object file: %s\n",
			filename);
		return { nullptr, bpf_object__close };
	}
	libbpf_object_ptr obj_ptr(obj, bpf_object__close);

	err = bpf_object__load(obj);
	if (err) {
		fprintf(stderr, "Error loading BPF object file: %s\n",
			filename);
		return { nullptr, bpf_object__close };
	}
	bpf_object__for_each_program(prog, obj)
	{
		// Link hasn't been used
		bpf_program__attach(prog);
		if (err) {
			fprintf(stderr, "Error attaching BPF program\n");
			return { nullptr, bpf_object__close };
		}
	}
	return obj_ptr;
}

int inject_agent(int target_pid, const char *agent_dynlib_path)
{
	// check the args
	frida_init();
	FridaInjector *injector = frida_injector_new();
	GError *error = NULL;
	guint id = frida_injector_inject_library_file_sync(injector, target_pid,
							   agent_dynlib_path,
							   "bpftime_agent_main",
							   "", NULL, &error);

	if (error != NULL) {
		fprintf(stderr, "%s\n", error->message);
		g_error_free(error);
		frida_unref(injector);
		frida_deinit();
		exit(EXIT_FAILURE);
	}

	printf("Successfully injected. ID: %u\n", id);

	frida_injector_close_sync(injector, NULL, NULL);
	frida_unref(injector);
	frida_deinit();
	return 0;
}

std::string get_lib_path(const char *library_name)
{
	struct stat st;
	auto so_path =
		std::string(DEFAULT_INSTALLATION_LOCATION) + library_name;
	if (stat(so_path.c_str(), &st) != 0) {
		cerr << "Error: necessary library " << so_path
		     << " not found:" << errno << endl;
		exit(1);
	}
	return so_path;
}

std::string get_agent_lib_path()
{
	return get_lib_path("/libbpftime-agent.so");
}

std::string get_server_lib_path()
{
	return get_lib_path("/libbpftime-syscall-server.so");
}

// Main program
int main(int argc, char *argv[])
{
	if (argc == 1) {
		cerr << "Usage: " << argv[0] << " [load|start|attach] ..."
		     << endl;
		return 1;
	}

	auto cmd = std::string(argv[1]);
	if (cmd == "load") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " load <EXECUTABLE>"
			     << endl;
			return 1;
		}
		auto so_path = get_server_lib_path();
		auto command_to_run =
			"LD_PRELOAD=" + so_path + " " + std::string(argv[2]);
		return system(command_to_run.c_str());
	} else if (cmd == "start") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " start <EXECUTABLE>"
			     << endl;
			return 1;
		}
		auto so_path = get_agent_lib_path();
		auto command_to_run =
			"LD_PRELOAD=" + so_path + " " + std::string(argv[2]);
		return system(command_to_run.c_str());
	} else if (cmd == "attach") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " attach <pid>" << endl;
			return 1;
		}
		// convert pid to int
		int pid = atoi(argv[2]);
		auto so_path = get_agent_lib_path();
		return inject_agent(pid, so_path.c_str());
	} else {
		cerr << "Invalid subcommand " << cmd << endl;
		return 1;
	}
	return EXIT_SUCCESS;
}