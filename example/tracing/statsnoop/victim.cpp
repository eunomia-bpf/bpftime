#include <cstdio>
#include <fstream>
#include <linux/limits.h>
#include <string>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
int main()
{
	std::vector<std::string> paths;
	std::ifstream ifs("/proc/mounts");
	while (ifs) {
		std::string line;
		std::getline(ifs, line);
		char path[PATH_MAX];
		sscanf(line.c_str(), "%*s%s", path);
		paths.push_back(path);
	}
	struct statfs fst;
	while (1) {
		for (const auto &p : paths) {
			puts("Do statfs");
			statfs(p.c_str(), &fst);
			usleep(1000 * 500);
		}
	}
	return 0;
}
