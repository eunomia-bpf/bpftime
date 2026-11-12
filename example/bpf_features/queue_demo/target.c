#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

__attribute__((noinline)) int target_function(int input_value, const char *msg)
{
	printf("target_function called with input=%d, msg=%s\n", input_value,
	       msg);
	return input_value * 2;
}

__attribute__((noinline)) void secondary_function(void)
{
	printf("secondary_function called\n");
	usleep(100000); // 100ms
}

int main()
{
	printf("Target program started, PID=%d\n", getpid());
	printf("Functions to be monitored:\n");
	printf("  - target_function: %p\n", target_function);
	printf("  - secondary_function: %p\n", secondary_function);
	printf("Starting periodic function calls...\n");

	int counter = 0;
	while (1) {
		counter++;

		char message[64];
		snprintf(message, sizeof(message), "call_%d", counter);
		int result = target_function(counter, message);

		if (counter % 5 == 0) {
			secondary_function();
		}

		printf("Main loop iteration %d completed, result=%d\n", counter,
		       result);

		sleep(1);

		if (counter % 10 == 0) {
			printf("--- %d iterations completed ---\n", counter);
		}
	}

	return 0;
}