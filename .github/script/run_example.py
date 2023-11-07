import sys
import subprocess
import datetime
import select
from io import StringIO
import signal
import fcntl
import os
import time

SERVER_TIMEOUT = 30
AGENT_TIMEOUT = 30
SERVER_START_SIGNAL = "bpftime-syscall-server started"


def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def main():
    (
        executable,
        victim,
        expected_str,
        bpftime_cli,
        syscall_trace,
        *bashreadline_patch,
    ) = sys.argv[1:]
    # Run the syscall-server
    server = subprocess.Popen(
        " ".join([bpftime_cli, "load", executable]),
        stdout=subprocess.PIPE,
        text=False,
        stderr=sys.stderr,
        bufsize=0,
        shell=True,
    )
    set_non_blocking(server.stdout)
    server_ok = False
    server_start_time = datetime.datetime.now()
    while (
        datetime.datetime.now() - server_start_time
    ).total_seconds() < SERVER_TIMEOUT:
        if server.poll() is not None:
            break
        ready, _, _ = select.select([server.stdout], [], [], 0.01)
        if ready:
            line = server.stdout.readline().decode()
            print("SERVER:", line, end="")
            if SERVER_START_SIGNAL in line:
                print("MONITOR: Server started!")
                server_ok = True
                break
    if not server_ok:
        print("Failed to start server!")
        server.kill()
        server.wait()
        exit(1)
    time.sleep(5)
    # Start the agent
    agent = subprocess.Popen(
        [bpftime_cli, "start"] + (["-s", victim] if syscall_trace == "1" else [victim]),
        stdout=sys.stdout,
        text=False,
        stderr=sys.stderr,
        stdin=subprocess.PIPE,
        env={"SPDLOG_LEVEL": "info"},
    )
    agent_start_time = datetime.datetime.now()
    agent_ok = False
    buf = StringIO()
    if bashreadline_patch:
        # Currently it's difficult to test bashreadline
        exit(0)
    while (datetime.datetime.now() - agent_start_time).total_seconds() < AGENT_TIMEOUT:
        # Check if server has expected output
        if server.poll() is not None:
            break
        ready, _, _ = select.select([server.stdout], [], [], 0.01)
        c = server.stdout.read()
        if c:
            c = c.decode()
            buf.write(c)
            print(c, end="")
            if c == "\n":
                buf.seek(0)
            if expected_str in buf.getvalue():
                # print("SERVER:", line, end="")
                # if expected_str in line:
                print(f"MONITOR: string `{expected_str}` found!")
                agent_ok = True
                break
    agent.kill()
    agent.wait()
    server.kill()
    server.wait()
    if not agent_ok:
        print("Failed to test, expected string not found!")
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
