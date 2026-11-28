/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * Replays structured inputs through the map-operation libFuzzer target in
 * runtime/fuzz, so the harness is built and exercised by every unit-test
 * run even without a libFuzzer toolchain.
 */
#include "platform_utils.hpp"
#include "catch2/catch_test_macros.hpp"
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
size_t bpftime_map_ops_fuzzer_last_value_buffer_size();

namespace
{

// Must match the input layout decoded by runtime/fuzz/map_ops_fuzzer.cpp.
enum shape_index : uint8_t {
	shape_hash,
	shape_array,
	shape_percpu_hash,
	shape_percpu_array,
	shape_lru_hash,
	shape_queue,
	shape_stack,
	shape_count,
};

enum keyed_op : uint8_t {
	op_update,
	op_lookup,
	op_delete,
	op_next_key,
	op_first_key,
	op_lookup_and_delete,
};

enum keyless_op : uint8_t {
	op_push,
	op_pop,
	op_peek,
	op_pop_lookup_and_delete,
};

constexpr uint8_t syscall_bit = 0x80;

struct input_builder {
	std::vector<uint8_t> bytes;
	size_t key_size = 0;
	size_t value_buf_size = 0;
	bool keyed = true;

	input_builder(uint8_t shape, uint8_t key_byte, uint8_t value_byte,
		      uint8_t max_entries_byte, uint8_t flags_byte)
	{
		bytes = { shape, key_byte, value_byte, max_entries_byte,
			  flags_byte };
		const size_t value_size = 1 + value_byte % 256;
		const bool percpu = shape == shape_percpu_hash ||
				    shape == shape_percpu_array;
		value_buf_size =
			percpu ? ((value_size + 7) & ~size_t(7)) *
					 bpftime_percpu_columns() :
				 value_size;
		if (shape == shape_array || shape == shape_percpu_array)
			key_size = sizeof(uint32_t);
		else if (shape == shape_queue || shape == shape_stack)
			keyed = false;
		else
			key_size = 1 + key_byte % 64;
	}

	void key_op(uint8_t op, uint8_t key_fill, uint8_t value_fill = 0)
	{
		bytes.push_back(op);
		for (size_t i = 0; i < key_size; i++)
			bytes.push_back(i == 0 ? key_fill : 0);
		if ((op & 0x7) == op_update)
			bytes.insert(bytes.end(), value_buf_size, value_fill);
	}

	void keyless_op(uint8_t op, uint8_t value_fill = 0)
	{
		bytes.push_back(op);
		if ((op & 0x3) == op_push)
			bytes.insert(bytes.end(), value_buf_size, value_fill);
	}
};

int run(const input_builder &input)
{
	const int ret =
		LLVMFuzzerTestOneInput(input.bytes.data(), input.bytes.size());
	// The encoded value layout only replays as intended if the fuzzer
	// consumed the same number of value bytes per write.
	REQUIRE(bpftime_map_ops_fuzzer_last_value_buffer_size() ==
		input.value_buf_size);
	return ret;
}

} // namespace

TEST_CASE("map ops fuzzer accepts degenerate inputs", "[fuzz]")
{
	REQUIRE(LLVMFuzzerTestOneInput(nullptr, 0) == 0);
	const uint8_t one = 0xff;
	REQUIRE(LLVMFuzzerTestOneInput(&one, 1) == 0);
	std::vector<uint8_t> all_ones(4096, 0xff);
	REQUIRE(LLVMFuzzerTestOneInput(all_ones.data(), all_ones.size()) == 0);
}

TEST_CASE("map ops fuzzer replays keyed map sequences", "[fuzz]")
{
	for (uint8_t shape : { shape_hash, shape_array, shape_percpu_hash,
			       shape_percpu_array, shape_lru_hash }) {
		for (uint8_t flags = 0; flags < 3; flags++) {
			for (uint8_t origin : { uint8_t(0), syscall_bit }) {
				// key size 8, value size 16, 4 entries.
				input_builder input(shape, 7, 15, 3, flags);
				for (uint8_t k = 0; k < 6; k++)
					input.key_op(origin | op_update, k, k);
				// BPF_NOEXIST, BPF_EXIST and an invalid flag.
				input.key_op(origin | 0x10 | op_update, 1, 9);
				input.key_op(origin | 0x20 | op_update, 9, 9);
				input.key_op(origin | 0x30 | op_update, 1, 9);
				input.key_op(origin | op_lookup, 1);
				input.key_op(origin | op_lookup, 9);
				input.key_op(origin | op_first_key, 0);
				input.key_op(origin | op_next_key, 0);
				input.key_op(origin | op_next_key, 3);
				input.key_op(origin | op_next_key, 200);
				input.key_op(origin | op_lookup_and_delete, 2);
				input.key_op(origin | op_delete, 1);
				input.key_op(origin | op_delete, 1);
				input.key_op(origin | op_lookup, 1);
				REQUIRE(run(input) == 0);
			}
		}
	}
}

TEST_CASE("map ops fuzzer replays queue and stack sequences", "[fuzz]")
{
	for (uint8_t shape : { shape_queue, shape_stack }) {
		// value size 8, 2 entries.
		input_builder input(shape, 0, 7, 1, 0);
		input.keyless_op(op_pop);
		input.keyless_op(op_peek);
		input.keyless_op(op_push, 1);
		input.keyless_op(op_push, 2);
		// Full: BPF_ANY fails, BPF_EXIST evicts, invalid flag fails.
		input.keyless_op(op_push, 3);
		input.keyless_op(0x20 | op_push, 4);
		input.keyless_op(0x30 | op_push, 5);
		input.keyless_op(syscall_bit | op_peek);
		input.keyless_op(op_pop_lookup_and_delete);
		input.keyless_op(syscall_bit | op_pop);
		input.keyless_op(op_pop);
		REQUIRE(run(input) == 0);
	}
}

TEST_CASE("map ops fuzzer covers maximal attribute sizes", "[fuzz]")
{
	for (uint8_t shape = 0; shape < shape_count; shape++) {
		// key size 64, value size 256, 64 entries.
		input_builder input(shape, 63, 255, 63, 0);
		for (int k = 0; k < 70; k++) {
			if (input.keyed)
				input.key_op(op_update, uint8_t(k), 0xab);
			else
				input.keyless_op(op_push, 0xab);
		}
		if (input.keyed) {
			input.key_op(syscall_bit | op_lookup, 10);
			input.key_op(syscall_bit | op_lookup_and_delete, 10);
			input.key_op(op_next_key, 63);
		} else {
			input.keyless_op(syscall_bit | op_pop);
		}
		REQUIRE(run(input) == 0);
	}
}
