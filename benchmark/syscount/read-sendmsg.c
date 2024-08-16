#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <time.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define NUM_ITERATIONS 1000000

void measure_read_time(int fd) {
    char buffer[1024];
    struct timespec start, end;
    long total_time_ns = 0;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        ssize_t bytes_read = read(fd, buffer, sizeof(buffer));
        clock_gettime(CLOCK_MONOTONIC, &end);

        if (bytes_read == -1) {
            perror("read");
            exit(EXIT_FAILURE);
        }

        long time_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        total_time_ns += time_ns;
    }

    printf("Average read() time: %ld ns\n", total_time_ns / NUM_ITERATIONS);
}

void measure_sendmsg_time(int sockfd) {
    struct msghdr msg;
    struct iovec iov;
    char buffer[1024] = "test message";
    struct timespec start, end;
    long total_time_ns = 0;

    memset(&msg, 0, sizeof(msg));
    iov.iov_base = buffer;
    iov.iov_len = sizeof(buffer);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(12345); // Arbitrary port number
    inet_pton(AF_INET, "127.0.0.1", &dest_addr.sin_addr); // Loopback address

    msg.msg_name = &dest_addr;
    msg.msg_namelen = sizeof(dest_addr);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        ssize_t bytes_sent = sendmsg(sockfd, &msg, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);

        if (bytes_sent == -1) {
            perror("sendmsg");
            exit(EXIT_FAILURE);
        }

        long time_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        total_time_ns += time_ns;
    }

    printf("Average sendmsg() time: %ld ns\n", total_time_ns / NUM_ITERATIONS);
}

int main() {
    // Open a file for reading
    int fd = open("./benchmark/syscount/testfile.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Create a socket for sendmsg()
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        perror("socket");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Measure read() time
    measure_read_time(fd);

    // Measure sendmsg() time
    measure_sendmsg_time(sockfd);

    close(fd);
    close(sockfd);
    return 0;
}
