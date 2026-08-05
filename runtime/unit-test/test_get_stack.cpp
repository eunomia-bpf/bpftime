#include "bpf_attach_ctx.hpp"
#include "linux/bpf.h"

#include <catch2/catch_test_macros.hpp>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

extern "C" int64_t bpftime_get_stack(uint64_t ctx_raw, uint64_t buf,
				     uint64_t size, uint64_t flags,
				     uint64_t unused);

namespace
{

class stack_attach_impl final : public bpftime::attach::base_attach_impl {
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
		if (name != "generate_stack") {
			return nullptr;
		}
		called = true;
		return new std::vector<uint64_t>{ 0x1234 };
	}

	bool called = false;
};

stack_attach_impl *test_attach_impl = nullptr;

} // namespace

bpftime::bpf_attach_ctx &get_global_attach_ctx()
{
	static bpftime::bpf_attach_ctx context;
	static bool registered = false;
	if (!registered) {
		auto impl = std::make_unique<stack_attach_impl>();
		test_attach_impl = impl.get();
		context.register_attach_impl(
			{ 6 }, std::move(impl), [](std::string_view, int &) {
				return std::unique_ptr<
					bpftime::attach::attach_private_data>();
			});
		registered = true;
	}
	return context;
}

TEST_CASE("get_stack accepts raw user stacks and rejects build IDs")
{
	uint64_t frame = 0;
	get_global_attach_ctx();
	REQUIRE(test_attach_impl != nullptr);

	test_attach_impl->called = false;
	REQUIRE(bpftime_get_stack(0, reinterpret_cast<uint64_t>(&frame),
				  sizeof(frame), BPF_F_USER_STACK, 0) >= 0);
	REQUIRE(test_attach_impl->called);

	test_attach_impl->called = false;
	REQUIRE(bpftime_get_stack(
			0, reinterpret_cast<uint64_t>(&frame), sizeof(frame),
			BPF_F_USER_STACK | BPF_F_USER_BUILD_ID, 0) == -ENOTSUP);
	REQUIRE_FALSE(test_attach_impl->called);
}
