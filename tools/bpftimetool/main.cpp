#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <memory>
#include <bpftime_shm.hpp>
#include <ostream>
#include <string>
#include <filesystem>

using namespace std;
using namespace bpftime;

// Main program
int main(int argc, char *argv[])
{
	if (argc == 1) {
		cerr << "Usage: " << argv[0] << " [load|import|export] ..."
		     << endl
		     << "Command-line tool to inspect and manage userspace eBPF objects"
		     << endl;
		return 1;
	}

	auto cmd = std::string(argv[1]);
	if (cmd == "load") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " load <fd> <JSON>"
			     << endl
			     << "Load a JSON file containing eBPF objects into the global shared memory"
			     << endl;
			return 1;
		}
		int fd = atoi(argv[2]);
		auto json_str = std::string(argv[3]);
		return bpftime_import_shm_handler_from_json(fd,
							    json_str.c_str());
	} else if (cmd == "export") {
		bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " export <filename>"
			     << endl
			     << "Export the global shared memory to a JSON file"
			     << endl;
			return 1;
		}
		auto filename = std::string(argv[2]);
		return bpftime_export_global_shm_to_json(filename.c_str());
	} else if (cmd == "import") {
		if (argc != 3) {
			cerr << "Usage: " << argv[0] << " import <filename>"
			     << endl
			     << "Import a JSON file containing eBPF objects into the global shared memory"
			     << endl;
			return 1;
		}
		auto filename = std::string(argv[2]);
		return bpftime_import_global_shm_from_json(filename.c_str());
	} else {
		cerr << "Invalid subcommand " << cmd << endl;
		return 1;
	}
	return EXIT_SUCCESS;
}