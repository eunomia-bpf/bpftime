#include "catch2/catch_message.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <pos/cuda_impl/utils/fatbin.h>
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static const char *FATBIN_FILE_PATH = TOSTRING(FATBIN_FILE);

TEST_CASE("Test fatbin parse")
{
	// POSUtil_CUDA_Fatbin parser;
	std::ifstream file(FATBIN_FILE_PATH, std::ios::binary);
	file.seekg(0, std::ios::end);
	auto file_size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(file_size);

	file.read(buffer.data(), file_size);

	INFO("file_size=" << file_size);
	std::vector<POSCudaFunctionDesp *> desp;
	std::map<std::string, POSCudaFunctionDesp *> cache;
	REQUIRE(POSUtil_CUDA_Fatbin::obtain_functions_from_cuda_binary(
			(uint8_t *)buffer.data(), buffer.size(), &desp,
			cache) == POS_SUCCESS);
	REQUIRE(true);
	REQUIRE(desp.size() >= 1);
}
