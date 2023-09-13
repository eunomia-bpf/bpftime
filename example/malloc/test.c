#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    putchar('i');
    printf("\nHello malloc!\n");
    while(1) {
        void *p = malloc(1024);
        printf("continue malloc...\n");
        sleep(1);
        free(p);
    }
    return 0;
}
