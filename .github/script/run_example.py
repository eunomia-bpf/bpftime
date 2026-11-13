import re
import sys
import asyncio
import typing
import signal
import os

SERVER_TIMEOUT = 30
AGENT_TIMEOUT = 30
SERVER_START_SIGNAL = "bpftime-syscall-server started"
CLEANUP_TIMEOUT = 10


def _expected_match(expected: str, line: str) -> bool:
    """Whether ``line`` satisfies ``expected``.

    The example matrix in the workflow passes plain substrings, but a few
    entries are written as regular expressions. Try the substring test
    first, then the pattern, and fall back to the substring result when the
    pattern does not compile: strings such as ``|*`` or ``mount(`` occur in
    the matrix and are not valid regular expressions, so letting the
    exception escape would abort the run for a plain-match entry.
    """
    if expected in line:
        return True
    try:
        return re.match(expected, line.strip()) is not None
    except re.error:
        return False


async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback_all: typing.List[typing.Tuple[asyncio.Event, str]] = [],
    check_error: bool = False,
):
    while True:
        t1 = asyncio.create_task(notify.wait())
        t2 = asyncio.create_task(stdout.readline())
        done, pending = await asyncio.wait(
            [t1, t2],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for item in pending:
            item.cancel()
        if t2 in done:
            stdout_raw = t2.result().decode()
            print(f"{title}:", stdout_raw, end="")
            for callback in callback_all:
                evt, sig = callback
                if _expected_match(sig, stdout_raw):
                    evt.set()
                    print("Callback triggered")
            if check_error and "[error]" in stdout_raw:
                assert False, "Error occurred in agent!"
        if t1 in done:
            break
        if stdout.at_eof():
            break


async def terminate_process(proc: asyncio.subprocess.Process, name: str):
    if proc is None:
        return
    if proc.returncode is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), CLEANUP_TIMEOUT)
        return
    except asyncio.TimeoutError:
        print(f"{name}: did not exit after SIGINT; sending SIGKILL")
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), CLEANUP_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"{name}: still did not exit after SIGKILL")


async def main():
    if len(sys.argv) < 6:
        print(
            "Usage: python script.py " \
            "<executable :str>" \
            "<victim :str> "\
            "<expected_str :str (regexp also works)> "\
            "<bpftime_cli :str> "\
            "<is_a_syscall :Union[0, 1]>"
        )
        return

    (
        executable,
        victim,
        expected_str,
        bpftime_cli,
        syscall_trace,
    ) = sys.argv[1:]
    server = None
    agent = None
    server_out = None
    agent_out = None
    try:
        bashreadline_patch = "readline" in executable
        should_exit = asyncio.Event()
        # Run the syscall-server
        server = await asyncio.subprocess.create_subprocess_exec(
            *(" ".join([bpftime_cli, "load", executable]).split()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        server_started_signal = asyncio.Event()
        expected_str_signal = asyncio.Event()

        server_out = asyncio.create_task(
            handle_stdout(
                server.stdout,
                should_exit,
                "SERVER",
                [
                    (server_started_signal, SERVER_START_SIGNAL),
                    (expected_str_signal, expected_str),
                ],
            )
        )

        await asyncio.wait_for(server_started_signal.wait(), SERVER_TIMEOUT)
        print("Server started!")

        # Start the agent
        agent_env = dict(os.environ)
        agent_env["SPDLOG_LEVEL"] = "debug"
        agent = await asyncio.subprocess.create_subprocess_exec(
            *(
                " ".join(
                    [bpftime_cli, "start"]
                    + (["-s", victim] if syscall_trace == "1" else [victim])
                ).split()
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=agent_env,
        )
        agent_out = asyncio.create_task(
            handle_stdout(agent.stdout, should_exit, "AGENT", [(expected_str_signal, expected_str)], True)
        )
        if bashreadline_patch:
            return
        await asyncio.wait_for(expected_str_signal.wait(), AGENT_TIMEOUT)
        print("Test successfully")
    finally:
        should_exit.set()
        try:
            out_tasks = [task for task in (server_out, agent_out) if task is not None]
            if out_tasks:
                await asyncio.gather(*out_tasks)
            await terminate_process(agent, "AGENT")
            await terminate_process(server, "SERVER")
        except Exception as ex:
            print(ex) 


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
