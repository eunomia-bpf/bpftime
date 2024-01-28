/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Test map operating helpers
#include "bpftime.hpp"
#include "map/map_common.h"
#include <assert.h>
#include <linux/bpf.h>
#include <cstdint>
#include <inttypes.h>
#include <stdio.h>
#include <map/map_hash.h>

// Remeber to change this
const char *obj_path = "hash-map-test.bpf.o";

uint64_t map_lookup(uint64_t map, uint64_t key, uint64_t _c, uint64_t _d,
		    uint64_t _e)
{
	struct Map *map_inst = bpftime_map_get_by_id((int32_t)map);
	assert(map_inst != NULL);
	return (uint64_t)map_inst->lookup_func(map_inst->map_impl,
					       (const void *)key);
}

uint64_t map_update(uint64_t map, uint64_t key, uint64_t val, uint64_t flags,
		    uint64_t _e)
{
	struct Map *map_inst = bpftime_map_get_by_id((int32_t)map);
	assert(map_inst != NULL);

	return (uint64_t)map_inst->update_func(map_inst->map_impl,
					       (const void *)key,
					       (const void *)val, flags);
}

uint64_t map_delete(uint64_t map, uint64_t key, uint64_t _c, uint64_t _d,
		    uint64_t _e)
{
	struct Map *map_inst = bpftime_map_get_by_id((int32_t)map);
	assert(map_inst != NULL);

	return (uint64_t)map_inst->delete_func(map_inst->map_impl,
					       (const void *)key);
}

int main(int argc, char **argv)
{
	int res = 1;
	struct bpftime_prog *ctx;
	uint64_t return_val;
	bpftime_map_init();
	int32_t map_id = bpftime_map_create(BPF_MAP_TYPE_HASH, 4, 4, 10, NULL);
	printf("Allocated map id: %" PRId32 "\n", map_id);
	assert(map_id == 0);
	ctx = bpftime_create_context();

	bpftime_register_helper(ctx, 1, "map_lookup", UFUNC_FN(map_lookup));
	bpftime_register_helper(ctx, 2, "map_update", UFUNC_FN(map_update));
	bpftime_register_helper(ctx, 3, "map_delete", UFUNC_FN(map_delete));

	res = bpftime_open_object(ctx, obj_path);
	if (res < 0) {
		printf("ebpf_open_object failed: %s", obj_path);
		return res;
	}
	// use the first program and relocate based on btf if btf has been
	// loaded
	res = bpftime_load_userspace(ctx, NULL, false);
	if (res < 0) {
		printf("ebpf_load_userspace failed\n");
		return res;
	}
	char mem[8];
	return_val = bpftime_exec_userspace(ctx, &mem, sizeof(mem));
	printf("return value: %" PRIu64 "\n", return_val);
	assert(return_val == 0);
	bpftime_free_context(ctx);
	return res;
}
