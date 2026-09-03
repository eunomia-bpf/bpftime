#include "bpf_attach_ctx.hpp"
#include "linux/bpf.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{

std::vector<uint64_t> mock_stack;

class mock_stack_attach_impl : public bpftime::attach::base_attach_impl {
    public:
	int detach_by_id(int) override
	{
		return 0;
	}

	int create_attach_with_ebpf_callback(
		bpftime::attach::ebpf_run_callback &&,
		const bpftime::attach::attach_private_data &, int) override
	{
		return 0;
	}

	void *call_attach_specific_function(const std::string &name,
					    void *) override
	{
		if (name != "generate_stack")
			return nullptr;
		return new std::vector<uint64_t>(mock_stack);
	}
};

bpftime::bpf_attach_ctx mock_attach_ctx;

} // namespace

bpftime::bpf_attach_ctx &get_global_attach_ctx()
{
	static const bool initialized = [] {
		mock_attach_ctx.register_attach_impl(
			{ 6 }, std::make_unique<mock_stack_attach_impl>(),
			[](const std::string_view &, int &) {
				return std::unique_ptr<
					bpftime::attach::attach_private_data>();
			});
		return true;
	}();
	(void)initialized;
	return mock_attach_ctx;
}

extern "C" int64_t bpftime_get_stack(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t);

TEST_CASE("get_stack rejects build IDs")
{
	REQUIRE(bpftime_get_stack(0, 0, 0,
				  BPF_F_USER_STACK | BPF_F_USER_BUILD_ID,
				  0) == -ENOTSUP);
}

TEST_CASE("get_stack uses byte-sized buffers and returns copied bytes")
{
	mock_stack = { 0x1111111111111111, 0x2222222222222222,
		       0x3333333333333333 };
	std::array<uint64_t, 4> buffer;
	buffer.fill(UINT64_MAX);

	auto copied =
		bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  sizeof(buffer), BPF_F_USER_STACK, 0);

	REQUIRE(copied == 3 * sizeof(uint64_t));
	REQUIRE(buffer[0] == mock_stack[0]);
	REQUIRE(buffer[1] == mock_stack[1]);
	REQUIRE(buffer[2] == mock_stack[2]);
	REQUIRE(buffer[3] == 0);
}

TEST_CASE("get_stack truncates only at complete frame boundaries")
{
	mock_stack = { 0x1111111111111111, 0x2222222222222222,
		       0x3333333333333333 };
	std::array<uint64_t, 3> buffer;
	buffer.fill(UINT64_MAX);

	auto copied =
		bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  2 * sizeof(uint64_t), BPF_F_USER_STACK, 0);

	REQUIRE(copied == 2 * sizeof(uint64_t));
	REQUIRE(buffer[0] == mock_stack[0]);
	REQUIRE(buffer[1] == mock_stack[1]);
	REQUIRE(buffer[2] == UINT64_MAX);
}

TEST_CASE("get_stack rejects a partial frame buffer")
{
	mock_stack = { 0x1111111111111111, 0x2222222222222222 };
	std::array<uint8_t, 15> buffer;
	buffer.fill(0xff);

	REQUIRE(bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  buffer.size(), BPF_F_USER_STACK,
				  0) == -EINVAL);
	REQUIRE(std::all_of(buffer.begin(), buffer.end(),
			    [](uint8_t value) { return value == 0; }));
}

TEST_CASE("get_stack applies frame skips before the byte limit")
{
	mock_stack = { 0x1111111111111111, 0x2222222222222222,
		       0x3333333333333333 };
	std::array<uint64_t, 3> buffer;
	buffer.fill(UINT64_MAX);

	auto copied =
		bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  sizeof(buffer), BPF_F_USER_STACK | 1, 0);

	REQUIRE(copied == 2 * sizeof(uint64_t));
	REQUIRE(buffer[0] == mock_stack[1]);
	REQUIRE(buffer[1] == mock_stack[2]);
	REQUIRE(buffer[2] == 0);
}

TEST_CASE("get_stack rejects skips beyond the available trace")
{
	mock_stack = { 0x1111111111111111, 0x2222222222222222 };
	std::array<uint64_t, 2> buffer;
	buffer.fill(UINT64_MAX);

	REQUIRE(bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  sizeof(buffer), BPF_F_USER_STACK | 3,
				  0) == -EFAULT);
	REQUIRE(std::all_of(buffer.begin(), buffer.end(),
			    [](uint64_t value) { return value == 0; }));
}
