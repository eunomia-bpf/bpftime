#ifndef TEST_DEF_BPFTIME_H
#define TEST_DEF_BPFTIME_H

#define CHECK_EXIT(ret)                                                        \
	if (ret != 0) {                                                        \
		fprintf(stderr, "Failed to load code: %s\n", errmsg);          \
		return -1;													 \
	}

#endif
