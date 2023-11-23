#include <dirent.h> /* Defines DT_* constants */
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>

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
	exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{
	return getdents64_main(argc, argv);
}
