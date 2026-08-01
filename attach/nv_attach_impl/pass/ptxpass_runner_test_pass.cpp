#include <cstdio>

__attribute__((constructor)) static void on_load()
{
	puts("constructor noise");
}

__attribute__((destructor)) static void on_unload()
{
	puts("destructor noise");
}

extern "C" void print_config(int length, char *output)
{
	puts("config noise");
	snprintf(output, length, R"({"mode":"config"})");
}

extern "C" int process_input(const char *input, int length, char *output)
{
	puts("process noise");
	snprintf(output, length, R"({"input":"%s"})", input);
	return 0;
}
