// SPDX-License-Identifier: MIT
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <unistd.h>

static volatile sig_atomic_t stop;
static void on_signal(int)
{
	stop = 1;
}

static int roundtrip(bpf_object *object, unsigned rounds)
{
	const int fd = bpf_object__find_map_fd_by_name(object, "mailbox");
	for (unsigned round = 1; round <= rounds && !stop; ++round) {
		// Exercise bits above 32 as well as changing the input each
		// round.
		const unsigned long long input = 0x100000000ULL + round * 17ULL;
		unsigned key = 0;
		if (bpf_map_update_elem(fd, &key, &input, BPF_ANY)) {
			std::perror("write mailbox request");
			return 1;
		}
		std::printf("CPU -> GPU round=%u input=%llu\n", round, input);
		std::fflush(stdout);
		const auto deadline = std::chrono::steady_clock::now() +
				      std::chrono::seconds(30);
		unsigned long long entry = 0, exit = 0;
		for (;;) {
			key = 1;
			if (bpf_map_lookup_elem(fd, &key, &entry)) {
				std::perror("read entry response");
				return 1;
			}
			key = 2;
			if (bpf_map_lookup_elem(fd, &key, &exit)) {
				std::perror("read return response");
				return 1;
			}
			if (entry == input * 3 + 1 && exit == input * 7 + 2)
				break;
			if (stop ||
			    std::chrono::steady_clock::now() >= deadline) {
				std::fprintf(
					stderr,
					"Round %u incomplete: entry=%llu (expected %llu), exit=%llu (expected %llu)\n",
					round, entry, input * 3 + 1, exit,
					input * 7 + 2);
				return 1;
			}
			usleep(1000);
		}
		std::printf("GPU -> CPU round=%u entry=%llu exit=%llu OK\n",
			    round, entry, exit);
		std::fflush(stdout);
	}
	if (stop)
		return 1;
	std::printf("ROUNDTRIP OK rounds=%u\n", rounds);
	std::fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	const std::string mode = argc > 1 ? argv[1] : "both";
	unsigned rounds = 3;
	if (mode == "roundtrip" && argc == 3) {
		char *end;
		const auto value = std::strtoul(argv[2], &end, 10);
		if (!argv[2][0] || *end || value < 1 || value > 1000)
			return 2;
		rounds = value;
	}
	if (argc > (mode == "roundtrip" ? 3 : 2) ||
	    (mode != "entry" && mode != "exit" && mode != "both" &&
	     mode != "roundtrip")) {
		std::fprintf(
			stderr,
			"Usage: %s [entry|exit|both|roundtrip [rounds:1-1000]]\n",
			argv[0]);
		return 2;
	}
	auto *object = bpf_object__open_file(BPFTIME_EXAMPLE_BPF, nullptr);
	if (!object || libbpf_get_error(object))
		return 1;
	bpf_program *program;
	bpf_object__for_each_program(program, object)
	{
		std::string name = bpf_program__name(program);
		bool selected =
			name == mode + "__rocm" ||
			(mode == "both" &&
			 (name == "entry__rocm" || name == "exit__rocm")) ||
			(mode == "roundtrip" &&
			 (name == "rt_entry__rocm" || name == "rt_exit__rocm"));
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
	if (mode == "roundtrip") {
		const int result = roundtrip(object, rounds);
		for (auto *link : links)
			bpf_link__destroy(link);
		bpf_object__close(object);
		return result;
	}
	const int map_fd = bpf_object__find_map_fd_by_name(object, "results");
	unsigned long long previous[5] = {};
	while (!stop) {
		unsigned long long values[5] = {};
		bool changed = false;
		for (unsigned key = 0; key < 5; ++key) {
			if (bpf_map_lookup_elem(map_fd, &key, &values[key])) {
				std::perror("read results map");
				stop = 1;
				break;
			}
			changed |= values[key] != previous[key];
			previous[key] = values[key];
		}
		if (changed) {
			std::printf(
				"Array: entry=%llu exit=%llu entry_value=%llu exit_value=%llu errors=%llu\n",
				values[0], values[1], values[2], values[3],
				values[4]);
			std::fflush(stdout);
		}
		usleep(100000);
	}
	for (auto *link : links)
		bpf_link__destroy(link);
	bpf_object__close(object);
	return 0;
}
