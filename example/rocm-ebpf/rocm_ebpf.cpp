// SPDX-License-Identifier: MIT
#include <bpf/libbpf.h>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>

static volatile sig_atomic_t stop;
static void on_signal(int)
{
	stop = 1;
}
int main(int argc, char **argv)
{
	const std::string mode = argc > 1 ? argv[1] : "both";
	if (argc > 2 || (mode != "entry" && mode != "exit" && mode != "both")) {
		std::fprintf(stderr, "Usage: %s [entry|exit|both]\n", argv[0]);
		return 2;
	}
	auto *object = bpf_object__open_file(BPFTIME_EXAMPLE_BPF, nullptr);
	if (!object || libbpf_get_error(object))
		return 1;
	bpf_program *program;
	bpf_object__for_each_program(program, object)
	{
		std::string name = bpf_program__name(program);
		bool selected = name == mode + "__rocm" ||
				(mode == "both" && (name == "entry__rocm" ||
						    name == "exit__rocm"));
		bpf_program__set_autoload(program, selected);
	}
	if (bpf_object__load(object)) {
		bpf_object__close(object);
		return 1;
	}
	std::vector<bpf_link *> links;
	bpf_object__for_each_program(program, object)
	{
		if (!bpf_program__autoload(program))
			continue;
		auto *link = bpf_program__attach(program);
		if (libbpf_get_error(link)) {
			for (auto *l : links)
				bpf_link__destroy(l);
			bpf_object__close(object);
			return 1;
		}
		links.push_back(link);
	}
	std::signal(SIGTERM, on_signal);
	std::signal(SIGINT, on_signal);
	std::printf("Loaded %s probes; Ctrl-C to detach.\n", mode.c_str());
	std::fflush(stdout);
	while (!stop)
		usleep(10000);
	for (auto *link : links)
		bpf_link__destroy(link);
	bpf_object__close(object);
	return 0;
}
