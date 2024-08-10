all: benchmark test
test: test.c
	gcc test.c -g -O3 -lpthread -o test

# test with the files in bpf-loader
TEST_CASES_DIRS=$(filter-out $(SKIP_TESTS),$(shell ls -l $(./) | grep ^d | awk '{print $$9}'))
benchmark: $(TEST_CASES_DIRS)

.PHONY:$(TEST_CASES_DIRS)
# build the test cases
$(TEST_CASES_DIRS):
	make -C $(TEST_EXAMPLE_DIR)$@/
