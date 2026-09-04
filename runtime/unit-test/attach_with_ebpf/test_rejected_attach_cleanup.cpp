/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_attach_ctx.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>

TEST_CASE("Rejected attach cleanup is fail-closed",
	  "[attach][cleanup][cpu]")
{
	constexpr int verifier_rejected = -4096;

	SECTION("successful cleanup permits state reset")
	{
		constexpr auto result =
			bpftime::detail::decide_rejected_attach_cleanup(
				verifier_rejected, 0);
		STATIC_REQUIRE(result.error_to_return == verifier_rejected);
		STATIC_REQUIRE(result.reset_state);
	}

	SECTION("failed cleanup retains tracking and reports cleanup error")
	{
		constexpr auto result =
			bpftime::detail::decide_rejected_attach_cleanup(
				verifier_rejected, -EBUSY);
		STATIC_REQUIRE(result.error_to_return == -EBUSY);
		STATIC_REQUIRE_FALSE(result.reset_state);
	}
}
