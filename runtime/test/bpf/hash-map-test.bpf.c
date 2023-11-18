/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
static void *(*bpf_map_lookup_elem)(void *map, const void *key) = (void *)1;
static long (*bpf_map_update_elem)(void *map, const void *key,
				   const void *value,
				   unsigned long flags) = (void *)2;
static long (*bpf_map_delete_elem)(void *map, const void *key) = (void *)3;

int bpf_main()
{
	void *MAP_ID = (void *)0;
	int a, b;
	a = 123;
	b = 234;
	bpf_map_update_elem(MAP_ID, &a, &b, 0);
	void *c = bpf_map_lookup_elem(MAP_ID, &a);
	if (*((int *)c) != 234)
		return -1;
	bpf_map_delete_elem(MAP_ID, &a);
	if (bpf_map_lookup_elem(MAP_ID, &a)) {
		return -2;
	}
	for (int i = 1; i <= 10; i++) {
		a = i;
		b = 1000 + i;
		bpf_map_update_elem(MAP_ID, &a, &b, 0);
	}
	return 0;
}
