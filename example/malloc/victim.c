#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    while(1) {
        void *p = malloc(1024);
        printf("continue malloc...\n");
        usleep(100 * 1000);
        free(p);
    }
    return 0;
}
