#include <frida_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace bpftime;
using namespace bpftime::attach;

#if defined(__x86_64__)
extern "C" __attribute__((naked, noinline)) uint64_t
__bpftime_test_tiny_attach_target(uint64_t a)
{
	asm volatile("lea 1(%rdi), %rax\n\t"
		     "ret\n\t");
}
#elif defined(__aarch64__)
extern "C" __attribute__((naked, noinline)) uint64_t
__bpftime_test_tiny_attach_target(uint64_t a)
{
	asm volatile("add x0, x0, #1\n\t"
		     "ret\n\t");
}
#endif

#if defined(__x86_64__) || defined(__aarch64__)
TEST_CASE("Test attach failure diagnostics for tiny function")
{
	frida_attach_impl man;

	try {
		int id = man.create_uprobe_at(
			(void *)__bpftime_test_tiny_attach_target,
			[](const pt_regs &) {});
		INFO("Frida accepted a tiny function on this platform, attach id="
		     << id);
		REQUIRE(man.detach_by_id(id) == 0);
	} catch (const std::runtime_error &ex) {
		std::string message = ex.what();
		INFO(message);
		REQUIRE(message.find("gum_interceptor_attach failed") !=
			std::string::npos);
		REQUIRE(message.find("attach_type=uprobe") !=
			std::string::npos);
		REQUIRE(message.find("first_bytes=") != std::string::npos);
		REQUIRE(message.find("Frida may reject very short functions") !=
			std::string::npos);
	}
}
#endif
