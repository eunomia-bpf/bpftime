/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

struct data {
	int key_size;
	int val_size;
} m;

static int (*map_look_up)(struct data *d, int key) = (void *) 1;

int func(void *mem) {
	int key = 3;
	int a= 5;
	int t = map_look_up(&m, key);
	return t + key;
}