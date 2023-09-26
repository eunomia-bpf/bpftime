#include <frida-core.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
extern "C" {
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
}

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

void inject_agent(cli_config &config)
{
	// check the args
	frida_init();
	FridaInjector *injector = frida_injector_new();
	GError *error = NULL;
	guint id = frida_injector_inject_library_file_sync(
		injector, config.target_pid, config.agent_dynlib_path.c_str(),
		"bpftime_agent_main", "", NULL, &error);

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
}


// Main program
int main(int argc, char *argv[])
{
	cli_config config;

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()("help,h", "Print help information")("version,v",
							       "Print version")(
		"dry-run", po::bool_switch(&config.dry_run),
		"dry run")("benchmark", "run the benchmark userspace function")(
		"kernel-uprobe", po::bool_switch(&config.kernel_uprobe),
		"Enable kernel uprobe")(
		"pid,p", po::value<int>(&config.pid)->default_value(0),
		"Process ID")("input-file",
			      po::value<std::string>(&config.input_file),
			      "Input file name");

	po::options_description hidden("Hidden options");
	hidden.add_options()("bpf-object-path",
			     po::value<std::string>(&config.bpf_object_path),
			     "BPF object file path");

	po::options_description cmdline_options;
	cmdline_options.add(desc).add(hidden);

	po::options_description visible_options;
	visible_options.add(desc);

	po::positional_options_description positionalOptions;
	positionalOptions.add("bpf-object-path", 1);

	std::cout << "start bpftime cli" << std::endl;

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv)
				  .options(cmdline_options)
				  .positional(positionalOptions)
				  .run(),
			  vm);
		po::notify(vm);
	} catch (const po::required_option &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	if (vm.count("help")) {
		std::cout << visible_options << std::endl;
		return 0;
	}

	if (vm.count("version")) {
		std::cout << "Version: " << version << std::endl;
		return 0;
	}

	if (!vm.count("bpf-object-path")) {
		std::cerr << "Error: The <bpf-object-path> option is required."
			  << std::endl;
		return 1;
	}

	return EXIT_SUCCESS;
}
