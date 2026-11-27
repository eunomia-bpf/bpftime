#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <memory>
#include <frida_attach_private_data.hpp>
#if defined(__linux__)
#include <frida_uprobe_attach_impl.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#endif
using namespace bpftime;

TEST_CASE("Test illegal parsing")
{
	SECTION("Test bad strings")
	{
		auto priv = std::make_unique<
			bpftime::attach::frida_attach_private_data>();
		REQUIRE(priv->initialize_from_string("aaa:") == -EINVAL);
	}
}

#if defined(__linux__)
static std::string read_self_maps()
{
	std::ifstream maps("/proc/self/maps");
	REQUIRE(maps.is_open());
	return { std::istreambuf_iterator<char>(maps),
		 std::istreambuf_iterator<char>() };
}

TEST_CASE(
	"Unmapped module resolution fails without force-loading the shared object")
{
	bpftime::attach::frida_attach_impl ctx;
	auto priv = std::make_unique<
		bpftime::attach::frida_attach_private_data>();
	REQUIRE(priv->initialize_from_string(
			"/bpftime-nonexistent-unmapped-module.so:16") ==
		-ENOENT);

	const std::string candidate = std::filesystem::canonical(
		BPFTIME_FRIDA_UNMAPPED_DUMMY_PATH).string();
	const std::string spec = candidate + ":16";
	REQUIRE(read_self_maps().find(candidate) == std::string::npos);

	REQUIRE(priv->initialize_from_string(spec) == -ENOENT);
	REQUIRE(read_self_maps().find(candidate) == std::string::npos);

	std::unique_ptr<void, decltype(&dlclose)> handle(
		dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL), &dlclose);
	REQUIRE(handle != nullptr);
	REQUIRE(priv->initialize_from_string(spec) == 0);
	REQUIRE(dlclose(handle.release()) == 0);
}
#endif
