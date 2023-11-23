#include <dirent.h> /* Defines DT_* constants */
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void start_timer(struct timespec *start_time)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, start_time);
}

void end_timer(struct timespec *end_time)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, end_time);
}

struct linux_dirent64 {
	ino64_t d_ino; /* 64-bit inode number */
	off64_t d_off; /* 64-bit offset to next structure */
	unsigned short d_reclen; /* Size of this dirent */
	unsigned char d_type; /* File type */
	char d_name[]; /* Filename (null-terminated) */
};

#define BUF_SIZE 2048

static double get_elapsed_time(struct timespec start_time,
			       struct timespec end_time)
{
	long seconds = end_time.tv_sec - start_time.tv_sec;
	long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
	if (start_time.tv_nsec > end_time.tv_nsec) { // clock underflow
		--seconds;
		nanoseconds += 1000000000;
	}
	return seconds * 1.0 + nanoseconds / 1000000000.0;
}

int getdents64_main(int argc, char *argv[])
{
	int fd;
	int nread;
	char buf[BUF_SIZE];
	struct linux_dirent64 *d;
	int bpos;
	char d_type;

	fd = open(argc > 1 ? argv[1] : ".", O_RDONLY | O_DIRECTORY);
	if (fd == -1) {
		perror("open");
		exit(EXIT_FAILURE);
	}

	for (;;) {
		nread = syscall(SYS_getdents64, fd, buf, BUF_SIZE);
		if (nread == -1) {
			perror("getdents64");
			exit(EXIT_FAILURE);
		}

		if (nread == 0)
			break;

		for (bpos = 0; bpos < nread;) {
			d = (struct linux_dirent64 *)(buf + bpos);
			printf("inode=%ld offset=%ld reclen=%u type=%d name=%s\n",
			       (long)d->d_ino, (long)d->d_off, d->d_reclen,
			       (int)d->d_type, d->d_name);
			bpos += d->d_reclen;
		}
	}
	int iter = 100000;
	struct timespec start_time, end_time;
	start_timer(&start_time);
	// calc time for 100*1000 getdents64
	for (int i = 0; i < iter; i++) {
		syscall(SYS_getdents64, fd, buf, BUF_SIZE);
	}
	end_timer(&end_time);
	double time = get_elapsed_time(start_time, end_time);
	printf("Average time usage %lf ns, iter %d times\n\n",
	       (time) / iter * 1000000000.0, iter);
	close(fd);
	return 0;
}

int newfsstatat_main(int argc, char *argv[])
{
	struct stat statbuf;
	int dfd; // Directory file descriptor
	int sflag = AT_SYMLINK_NOFOLLOW; // Do not follow symbolic links

	// Open the directory to get a file descriptor for it
	dfd = open(argc > 1 ? argv[1] : ".", O_RDONLY | O_DIRECTORY);
	if (dfd == -1) {
		perror("open");
		exit(EXIT_FAILURE);
	}
	const char *file_name = argc > 2 ? argv[2] : ".profile";
	// Use fstatat to get information about a file within that directory
	if (fstatat(dfd, file_name, &statbuf, sflag) == -1) {
		perror("newfstatat");
		close(dfd);
		exit(EXIT_FAILURE);
	}

	// Output some of the file information
	printf("File size: %lld bytes\n", (long long)statbuf.st_size);
	printf("File inode: %ld\n", (long)statbuf.st_ino);

	int iter = 100000;
	struct timespec start_time, end_time;
	start_timer(&start_time);
	// calc time for 100*1000 getdents64
	for (int i = 0; i < iter; i++) {
		fstatat(dfd, file_name, &statbuf, sflag);
	}
	end_timer(&end_time);
	double time = get_elapsed_time(start_time, end_time);
	printf("Average time usage %lf ns, iter %d times\n\n",
	       (time) / iter * 1000000000.0, iter);
	// Close the directory file descriptor
	close(dfd);

	return 0;
}

int main(int argc, char *argv[])
{
	if (argc <= 1 || strcmp(argv[1], "--help") == 0) {
		printf("Usage: %s [getdents64|newfsstatat]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	if (strcmp(argv[1], "getdents64") == 0) {
		return getdents64_main(argc - 1, argv + 1);
	} else if (strcmp(argv[1], "newfsstatat") == 0) {
		return newfsstatat_main(argc - 1, argv + 1);
	} else {
		printf("Usage: %s [getdents64|newfsstatat]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}
