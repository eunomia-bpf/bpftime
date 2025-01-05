#include "catch2/catch_message.hpp"
#include "catch2/internal/catch_stdstreams.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <ostream>
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
	std::vector<std::string> ptx_out;
	REQUIRE(POSUtil_CUDA_Fatbin::obtain_functions_from_cuda_binary(
			(uint8_t *)buffer.data(), buffer.size(), &desp, cache,
			ptx_out) == POS_SUCCESS);
	REQUIRE(true);
	REQUIRE(desp.size() >= 1);
	INFO("Got ptx count" << ptx_out.size());
	for (const auto &ptx : ptx_out) {
		std::cout << "PTX " << ptx << std::endl;
	}
	REQUIRE(false);
}
