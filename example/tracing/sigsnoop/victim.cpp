#include <signal.h>
#include <unistd.h>

int main()
{
	int self_pid = getpid();
	kill(self_pid, 0);
	kill(self_pid, SIGCHLD);
	killpg(self_pid, 0);
	kill(-1, 0);
	return 0;
}
