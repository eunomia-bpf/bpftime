#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <memory>
#include <frida_attach_private_data.hpp>
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
