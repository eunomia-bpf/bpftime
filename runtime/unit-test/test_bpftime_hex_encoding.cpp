/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_hex_encoding.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>

// Keep allocation injection out of the runtime and Catch2 executables.
// This test links only the C++ standard library and the pure encoding helper.
static thread_local int fail_after = -1;
static thread_local unsigned failures = 0;

void *operator new(std::size_t size)
{
	if (fail_after >= 0 && fail_after-- == 0) {
		fail_after = -1;
		++failures;
		throw std::bad_alloc();
	}
	if (void *p = std::malloc(size ? size : 1))
		return p;
	throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

int main()
{
	const unsigned char edge_bytes[] = {0, 1, 0x7f, 0x80, 0xff};
	if (bpftime::buffer_to_hex_string(nullptr, 0) != "" ||
	    bpftime::buffer_to_hex_string(edge_bytes, sizeof(edge_bytes)) !=
		    "00017f80ff")
		return 1;
	try {
		bpftime::buffer_to_hex_string(nullptr, std::string().max_size() / 2 + 1);
		return 2;
	} catch (const std::length_error &) {
	}

	std::array<unsigned char, 1024> bytes{};
	for (size_t i = 0; i < bytes.size(); ++i)
		bytes[i] = static_cast<unsigned char>(i);
	const auto expected = bpftime::buffer_to_hex_string(bytes.data(), bytes.size());
	constexpr char digits[] = "0123456789abcdef";
	if (expected.size() != bytes.size() * 2)
		return 3;
	for (size_t i = 0; i < bytes.size(); ++i)
		if (expected[2 * i] != digits[bytes[i] >> 4] ||
		    expected[2 * i + 1] != digits[bytes[i] & 0xf])
			return 4;

	unsigned injected = 0;
	for (int allocation = 0; allocation != 8; ++allocation) {
		const unsigned before = failures;
		bool threw = false;
		bool complete = false;
		fail_after = allocation;
		try {
			const auto encoded =
				bpftime::buffer_to_hex_string(bytes.data(), bytes.size());
			complete = encoded == expected;
		} catch (const std::bad_alloc &) {
			threw = true;
		}
		fail_after = -1;
		const bool fired = failures != before;
		injected += fired;
		if ((fired && !threw) || (!fired && !complete)) {
			std::fprintf(stderr,
				     "hex encoding lost allocation failure at index %d\n",
				     allocation);
			return 5;
		}
	}
	if (injected == 0)
		return 6;
	std::printf("hex encoding: exact bytes, size guard, and %u allocation failure(s) passed\n",
		    injected);
	return 0;
}
