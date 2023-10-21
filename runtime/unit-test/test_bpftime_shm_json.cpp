#include "catch2/catch_message.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/per_cpu_array_map.hpp>
#include <bpftime_shm_json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "common_def.hpp"
using namespace boost::interprocess;
using namespace bpftime;

TEST_CASE("Test bufferToHexString function")
{
    unsigned char buffer[] = {0x12, 0x34, 0x56, 0x78};
    std::string expected = "12345678";
    std::string result = bufferToHexString(buffer, sizeof(buffer));
    REQUIRE(result == expected);
}

TEST_CASE("Test hexStringToBuffer function")
{
    std::string hexString = "12345678";
    unsigned char expected[] = {0x12, 0x34, 0x56, 0x78};
    unsigned char buffer[sizeof(expected)];
    int result = hexStringToBuffer(hexString, buffer, sizeof(buffer));
    REQUIRE(result == 0);
    REQUIRE(std::memcmp(buffer, expected, sizeof(buffer)) == 0);
}

TEST_CASE("Test hexStringToBuffer function with invalid input")
{
    std::string hexString = "1234567";
    unsigned char buffer[4];
    int result = hexStringToBuffer(hexString, buffer, sizeof(buffer));
    REQUIRE(result == -1);
}