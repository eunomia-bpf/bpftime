#include "cricket-types.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cricket-util.h"


void cricket_error_unreachable(void)
{
    printf("ERROR 2200: We've reached an unreachable state. Anything is possible. The limits were in our heads all along. Follow your dreams.\n");
}
double time_diff_sec(const struct timeval *tv1, const struct timeval *tv2)
{
    return fabs((tv2->tv_sec - tv1->tv_sec) +
                ((tv2->tv_usec - tv1->tv_usec) / 1000000.0));
}

uint time_diff_usec(const struct timeval *tv1, const struct timeval *tv2)
{
    return abs((tv2->tv_sec - tv1->tv_sec) * 1000000 + tv2->tv_usec -
               tv1->tv_usec);
}

void print_binary32(uint32_t num)
{
    uint8_t i;
    for (i = 0; i != 32; ++i) {
        if (num & (1LLU << 31))
            printf("1");
        else
            printf("0");
        num <<= 1;
    }
    printf("\n");
}

void print_binary64(uint64_t num)
{
    uint8_t i;
    for (i = 0; i != 64; ++i) {
        if (num & (1LLU << 63))
            printf("1");
        else
            printf("0");
        num <<= 1;
    }
    printf("\n");
}

