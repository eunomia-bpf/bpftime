/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_JSON_EXPORTER_HPP
#define BPFTIME_JSON_EXPORTER_HPP

#include <string>
#include <sstream>
#include <stdio.h>
#include <iomanip>
#include "bpftime_shm_internal.hpp"

namespace bpftime
{
static inline std::string buffer_to_hex_string(const unsigned char *buffer,
					    size_t bufferSize)
{
	std::stringstream ss;
	ss << std::hex << std::setfill('0');
	for (size_t i = 0; i < bufferSize; i++) {
		ss << std::setw(2) << static_cast<int>(buffer[i]);
	}
	return ss.str();
}

static inline int hex_string_to_buffer(const std::string &hexString,
				     unsigned char *buffer, size_t bufferSize)
{
	if (hexString.size() % 2 || hexString.size() / 2 != bufferSize ||
	    (bufferSize && buffer == nullptr))
		return -1;
	auto digit = [](char c) -> int {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		return -1;
	};
	// Validate first, so invalid input never partially changes the buffer.
	for (char c : hexString)
		if (digit(c) < 0) return -1;
	for (size_t i = 0; i < bufferSize; ++i)
		buffer[i] = static_cast<unsigned char>(
			(digit(hexString[2 * i]) << 4) |
			digit(hexString[2 * i + 1]));
	return 0;
}

int bpftime_export_shm_to_json(const bpftime_shm& shm, const char *filename);
int bpftime_import_shm_from_json(bpftime_shm& shm, const char *filename);
int bpftime_import_shm_handler_from_json(bpftime_shm &shm, int fd,
					  const char *json_string);
} // namespace bpftime
#endif // BPFTIME_JSON_EXPORTER_HPP
