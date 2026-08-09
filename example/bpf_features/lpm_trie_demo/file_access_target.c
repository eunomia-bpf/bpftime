// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* bpftime LPM Trie test target program */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdbool.h>

// Test functions - these will be hooked by eBPF program
int test_file_access(const char *filename, int flags)
{
	printf("Testing file access: %s (flags=0x%x)\n", filename, flags);

	// This function will be intercepted by uprobe
	// The BPF program will check LPM Trie and set access policy
	// For now, we simulate the check by always allowing access
	// The actual access control should be implemented in BPF program

	// Try to open file
	int fd;
	if (flags & O_CREAT) {
		// If creating file, provide mode parameter
		fd = open(filename, flags, 0644);
	} else {
		fd = open(filename, flags);
	}

	if (fd >= 0) {
		close(fd);
		return 0; // Success
	} else {
		return -1; // Failure
	}
}

int test_file_access_at(const char *filename, int flags)
{
	printf("Testing file access (at): %s (flags=0x%x)\n", filename, flags);

	// Try to open file
	int fd;
	if (flags & O_CREAT) {
		// If creating file, provide mode parameter
		fd = openat(AT_FDCWD, filename, flags, 0644);
	} else {
		fd = openat(AT_FDCWD, filename, flags);
	}

	if (fd >= 0) {
		close(fd);
		return 0; // Success
	} else {
		return -1; // Failure
	}
}

// Test case structure
struct test_case {
	const char *path;
	const char *description;
	int flags;
	int expected_result; // 1=should succeed, 0=should fail
};

int main(int argc, char **argv)
{
	printf("bpftime LPM Trie file access test program\n");
	printf("=========================================\n\n");

	// Create test files
	printf("Creating test files...\n");
	if (system("mkdir -p /tmp/test_allowed") != 0) {
		printf("Warning: Failed to create directory\n");
	}
	if (system("echo 'test content' > /tmp/test_allowed/file1.txt") != 0) {
		printf("Warning: Failed to create file1\n");
	}
	if (system("echo 'test content' > /tmp/test_allowed/file2.txt") != 0) {
		printf("Warning: Failed to create file2\n");
	}
	printf("Test files created\n\n");

	// Test cases
	struct test_case test_cases[] = {
		// Should be allowed file accesses (matching LPM Trie prefixes)
		{ "/tmp/test_allowed/file1.txt", "temp file 1", O_RDONLY, 1 },
		{ "/tmp/test_allowed/file2.txt", "temp file 2", O_RDONLY, 1 },
		{ "/tmp/new_file.txt", "new temp file", O_CREAT | O_WRONLY, 1 },

		// Should be denied file accesses (not matching any LPM Trie
		// prefix)
		{ "/opt/restricted_file.txt", "restricted opt file", O_RDONLY,
		  0 },
		{ "/var/log/restricted.log", "restricted log file", O_RDONLY,
		  0 },
		{ "/root/secret.txt", "root directory file", O_RDONLY, 0 },
	};

	int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
	int passed = 0;
	int failed = 0;

	printf("Starting file access tests...\n");
	printf("Note: eBPF program will intercept these accesses and apply LPM Trie rules\n\n");

	for (int i = 0; i < num_tests; i++) {
		printf("Test %d/%d: %s\n", i + 1, num_tests,
		       test_cases[i].description);
		printf("   Path: %s\n", test_cases[i].path);

		// Call test function - this will trigger eBPF uprobe
		int result = test_file_access(test_cases[i].path,
					      test_cases[i].flags);

		bool success = (result == 0);
		bool expected = (test_cases[i].expected_result == 1);

		if (success == expected) {
			printf("   Result: %s (as expected)\n",
			       success ? "SUCCESS" : "FAILED");
			passed++;
		} else {
			printf("   Result: %s (unexpected, expected: %s)\n",
			       success ? "SUCCESS" : "FAILED",
			       expected ? "SUCCESS" : "FAILED");
			failed++;
		}

		printf("\n");

		// Brief delay to let monitor process events
		usleep(500000); // 500ms
	}

	printf("Test results summary:\n");
	printf("====================\n");
	printf("Total tests: %d\n", num_tests);
	printf("Passed: %d\n", passed);
	printf("Failed: %d\n", failed);
	printf("Success rate: %.1f%%\n", (float)passed / num_tests * 100);

	if (passed == num_tests) {
		printf("\nAll tests passed! bpftime LPM Trie working correctly\n");
		return 0;
	} else {
		printf("\nSome tests failed, please check eBPF program and LPM Trie configuration\n");
		return 1;
	}
}