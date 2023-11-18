/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
const int i = 0;
static int j = 1;
int k = 1;
typedef unsigned long long uint64_t;
struct data {
	uint64_t context;
};

int add1(struct data *d, int sz) {
 	return i + j;
}

int add2(struct data *d, int sz) {
 	return i + k;
}

int add3(struct data *d, int sz) {
 	return j + k;
}
