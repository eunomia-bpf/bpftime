#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include "linux/bpf.h"

#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

std::vector<uint64_t> mock_stack;
// Set while a test plays a probe with no unwinding context: the attach
// implementation has no Gum invocation and therefore no frames to return.
bool mock_outside_hook = false;

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
					    void *data) override
	{
		auto *out =
			static_cast<bpftime::attach::stack_walk_data *>(data);
		if (out == nullptr)
			return nullptr;
		out->count = 0;
		if (name != "generate_stack" || mock_outside_hook)
			return nullptr;
		out->count = std::min(
			mock_stack.size(),
			bpftime::attach::stack_walk_data::max_frames);
		std::copy_n(mock_stack.data(), out->count, out->frames);
		return out->count > 0 ? out : nullptr;
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
extern "C" int64_t bpftime_get_stackid(uint64_t, uint64_t, uint64_t, uint64_t,
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

TEST_CASE("get_stack outside a hook reports an empty stack without logging")
{
	// A syscall-trace or tracepoint probe has a synthesized pt_regs and no
	// Gum invocation, so the attach implementation returns no frames. The
	// kernel reports that as an empty stack, and the helper must not turn
	// it into an error a BPF program would have to special-case, nor log at
	// error level for a program that runs outside a hook by design.
	std::ostringstream output;
	auto previous = spdlog::default_logger();
	auto logger = std::make_shared<spdlog::logger>(
		"get_stack_outside_hook",
		std::make_shared<spdlog::sinks::ostream_sink_mt>(output));
	logger->set_level(spdlog::level::err);
	spdlog::set_default_logger(logger);
	mock_outside_hook = true;
	std::array<uint64_t, 2> buffer;
	buffer.fill(UINT64_MAX);

	auto copied =
		bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  sizeof(buffer), BPF_F_USER_STACK, 0);

	mock_outside_hook = false;
	spdlog::set_default_logger(previous);

	REQUIRE(copied == 0);
	REQUIRE(buffer[0] == 0);
	REQUIRE(buffer[1] == 0);
	REQUIRE(output.str().empty());
}

TEST_CASE("get_stack and get_stackid populate caller stack_walk_data in place")
{
	const std::string shm_name =
		"bpftime_test_stackid_in_place_" + std::to_string(getpid());
	const char *old_env = getenv("BPFTIME_GLOBAL_SHM_NAME");
	const std::string saved_env = old_env ? old_env : "";
	const bool had_env = old_env != nullptr;
	int map_fd = -1;
	struct cleanup_guard {
		int &map_fd;
		bool had_env;
		const std::string &saved_env;
		~cleanup_guard()
		{
			mock_outside_hook = false;
			if (map_fd >= 0)
				bpftime_close(map_fd);
			bpftime_destroy_global_shm();
			bpftime_remove_global_shm();
			if (had_env)
				setenv("BPFTIME_GLOBAL_SHM_NAME",
				       saved_env.c_str(), 1);
			else
				unsetenv("BPFTIME_GLOBAL_SHM_NAME");
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
	} guard{ map_fd, had_env, saved_env };

	setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	bpftime::bpf_map_attr attr{};
	attr.type =
		static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_STACK_TRACE);
	attr.key_size = sizeof(uint32_t);
	attr.value_size = 8 * sizeof(uint64_t);
	attr.max_ents = 64;
	map_fd = bpftime_maps_create(-1, "stack_in_place", attr);
	REQUIRE(map_fd >= 0);

	mock_stack = { 0xaaaa0001ULL, 0xbbbb0002ULL, 0xcccc0003ULL };

	std::array<uint64_t, 4> buffer{};
	REQUIRE(bpftime_get_stack(0, reinterpret_cast<uint64_t>(buffer.data()),
				  sizeof(buffer), BPF_F_USER_STACK | 1,
				  0) == 2 * sizeof(uint64_t));
	REQUIRE(buffer[0] == 0xbbbb0002ULL);
	REQUIRE(buffer[1] == 0xcccc0003ULL);

	int64_t sid = bpftime_get_stackid(0, static_cast<uint64_t>(map_fd),
					  BPF_F_USER_STACK | 1, 0, 0);
	REQUIRE(sid >= 0);

	mock_outside_hook = true;
	int64_t empty_sid = bpftime_get_stackid(
		0, static_cast<uint64_t>(map_fd),
		BPF_F_USER_STACK | BPF_F_REUSE_STACKID, 0, 0);
	REQUIRE(empty_sid >= 0);
}
