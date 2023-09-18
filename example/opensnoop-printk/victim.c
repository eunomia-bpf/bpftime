#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
int main()
{
    putchar('\n');
	while (1) {
		puts("Opening test.txt..");
		int fd = open("test.txt", O_CREAT | O_RDWR);
		printf("Get fd %d\n", fd);
		usleep(500 * 1000);
		puts("Closing fd..");
		close(fd);
	}
	return 0;
}
