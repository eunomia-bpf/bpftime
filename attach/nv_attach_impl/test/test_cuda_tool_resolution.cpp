#include "catch2/catch_test_macros.hpp"
#include "nv_attach_impl.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace bpftime::attach;

namespace
{

class scoped_env_var {
    public:
	scoped_env_var(const char *name, const char *value) : name(name)
	{
		if (const char *current = std::getenv(name); current != nullptr) {
			previous_value = current;
		}
		if (value != nullptr) {
			setenv(name, value, 1);
		} else {
			unsetenv(name);
		}
	}

	~scoped_env_var()
	{
		if (previous_value.has_value()) {
			setenv(name.c_str(), previous_value->c_str(), 1);
		} else {
			unsetenv(name.c_str());
		}
	}

    private:
	std::string name;
	std::optional<std::string> previous_value;
};

class scoped_temp_dir {
    public:
	scoped_temp_dir()
	{
		std::array<char, 64> tmpl{};
		std::snprintf(tmpl.data(), tmpl.size(),
			      "/tmp/bpftime-cuda-tool.XXXXXX");
		auto *created = mkdtemp(tmpl.data());
		if (created == nullptr)
			throw std::runtime_error("mkdtemp failed");
		dir_path = created;
	}

	~scoped_temp_dir()
	{
		std::error_code ec;
		std::filesystem::remove_all(dir_path, ec);
	}

	std::filesystem::path path() const
	{
		return dir_path;
	}

    private:
	std::filesystem::path dir_path;
};

std::filesystem::path create_executable_tool(const std::filesystem::path &root,
					     const std::string &tool_name)
{
	auto bin_dir = root / "bin";
	std::filesystem::create_directories(bin_dir);
	auto tool_path = bin_dir / tool_name;
	std::ofstream ofs(tool_path);
	ofs << "#!/bin/sh\nexit 0\n";
	ofs.close();
	chmod(tool_path.c_str(), 0755);
	return tool_path;
}

} // namespace

TEST_CASE("resolve_cuda_tool_path prefers BPFTIME_CUDA_ROOT",
	  "[nv_attach_impl]")
{
	scoped_temp_dir preferred_root;
	scoped_temp_dir fallback_root;
	auto expected = create_executable_tool(preferred_root.path(),
					       "cuobjdump");
	create_executable_tool(fallback_root.path(), "cuobjdump");

	scoped_env_var bpftime_cuda_root("BPFTIME_CUDA_ROOT",
					 preferred_root.path().c_str());
	scoped_env_var cuda_home("CUDA_HOME", fallback_root.path().c_str());
	scoped_env_var cuda_path("CUDA_PATH", nullptr);
	scoped_env_var path("PATH", "");

	auto resolved = resolve_cuda_tool_path("cuobjdump");
	REQUIRE(resolved.has_value());
	REQUIRE(resolved->lexically_normal() == expected.lexically_normal());
}

TEST_CASE("resolve_cuda_tool_path falls back to PATH", "[nv_attach_impl]")
{
	scoped_temp_dir path_root;
	auto expected = create_executable_tool(path_root.path(), "cuobjdump");

	scoped_env_var bpftime_cuda_root("BPFTIME_CUDA_ROOT", nullptr);
	scoped_env_var cuda_home("CUDA_HOME", nullptr);
	scoped_env_var cuda_path("CUDA_PATH", nullptr);
	scoped_env_var path("PATH", (path_root.path() / "bin").c_str());

	auto resolved = resolve_cuda_tool_path("cuobjdump");
	REQUIRE(resolved.has_value());
	REQUIRE(resolved->lexically_normal() == expected.lexically_normal());
}

TEST_CASE("resolve_cuda_tool_path returns empty for missing tools",
	  "[nv_attach_impl]")
{
	scoped_env_var bpftime_cuda_root("BPFTIME_CUDA_ROOT", nullptr);
	scoped_env_var cuda_home("CUDA_HOME", nullptr);
	scoped_env_var cuda_path("CUDA_PATH", nullptr);
	scoped_env_var path("PATH", "");

	auto resolved = resolve_cuda_tool_path("bpftime-missing-cuda-tool");
	REQUIRE_FALSE(resolved.has_value());
}
