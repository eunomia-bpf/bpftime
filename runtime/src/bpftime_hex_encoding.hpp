/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_HEX_ENCODING_HPP
#define BPFTIME_HEX_ENCODING_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

namespace bpftime
{
static inline std::string buffer_to_hex_string(const unsigned char *buffer,
                                               std::size_t bufferSize)
{
	std::string result;
	if (bufferSize > result.max_size() / 2)
		throw std::length_error("hex encoding exceeds string size limit");
	// Allocate before encoding. Stream insertion can swallow bad_alloc into
	// badbit and return a truncated string; export must instead fail atomically.
	result.resize(bufferSize * 2);
	constexpr char digits[] = "0123456789abcdef";
	for (std::size_t i = 0; i < bufferSize; ++i) {
		result[2 * i] = digits[buffer[i] >> 4];
		result[2 * i + 1] = digits[buffer[i] & 0xf];
	}
	return result;
}
} // namespace bpftime
#endif
