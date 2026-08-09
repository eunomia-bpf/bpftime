#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Simulate user access function
__attribute__((noinline)) int user_access(int user_id, const char *resource)
{
	printf("user_access called: user_id=%d, resource=%s\n", user_id,
	       resource);

	// Simulate some processing time
	usleep(50000); // 50ms

	return user_id + strlen(resource);
}

// Simulate admin operation function
__attribute__((noinline)) void admin_operation(int admin_id,
					       const char *operation)
{
	printf("admin_operation called: admin_id=%d, operation=%s\n", admin_id,
	       operation);
	usleep(100000); // 100ms
}

// Simulate system event function
__attribute__((noinline)) void system_event(const char *event_type)
{
	printf("system_event called: event_type=%s\n", event_type);
	usleep(25000); // 25ms
}

int main()
{
	printf("Bloom Filter Demo Target Program started, PID=%d\n", getpid());
	printf("Functions to be monitored:\n");
	printf("  - user_access: %p\n", user_access);
	printf("  - admin_operation: %p\n", admin_operation);
	printf("  - system_event: %p\n", system_event);
	printf("Starting simulation...\n\n");

	// Predefined user IDs and resources
	int user_ids[] = { 1001, 1002, 1003, 1004, 1005,
			   1006, 1007, 1008, 1009, 1010 };
	const char *resources[] = { "file1.txt", "file2.txt", "database",
				    "config",	 "logs",      "backup" };
	const char *admin_ops[] = { "create_user", "delete_file",
				    "backup_system", "update_config" };
	const char *events[] = { "login", "logout", "error", "warning",
				 "info" };

	int counter = 0;
	while (1) {
		counter++;

		// Simulate user access (high frequency)
		if (counter % 2 == 0) {
			int user_id = user_ids[counter % 10];
			const char *resource = resources[counter % 6];
			user_access(user_id, resource);
		}

		// Simulate admin operations (medium frequency)
		if (counter % 5 == 0) {
			int admin_id = 2000 + (counter % 3);
			const char *operation = admin_ops[counter % 4];
			admin_operation(admin_id, operation);
		}

		// Simulate system events (low frequency)
		if (counter % 3 == 0) {
			const char *event = events[counter % 5];
			system_event(event);
		}

		printf("--- Iteration %d completed ---\n\n", counter);

		sleep(1);

		// Display statistics every 10 iterations
		if (counter % 10 == 0) {
			printf("=== %d iterations completed ===\n", counter);
		}
	}

	return 0;
}