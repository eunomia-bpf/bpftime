import sys
import asyncio
import typing
import signal
import os

SERVER_TIMEOUT = 30
AGENT_TIMEOUT = 30
SERVER_START_SIGNAL = "bpftime-syscall-server started"
# How long to wait for a child that printed the expected output to exit on its
# own before deciding it is the usual long-lived victim.
EXIT_GRACE = 5
CLEANUP_TIMEOUT = 10


async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback_all: typing.List[typing.Tuple[asyncio.Event, str]] = [],
    check_error: bool = False,
    errors: typing.Optional[typing.List[str]] = None,
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
            s = t2.result().decode()
            print(f"{title}:", s, end="")
            for callback in callback_all:
                evt, sig = callback
                if sig in s:
                    evt.set()
                    print("Callback triggered")
            if check_error and "[error]" in s and errors is not None:
                errors.append(f"{title}: printed an error line: {s.strip()}")
        if t1 in done:
            break
        if stdout.at_eof():
            break


async def exited_early(proc: asyncio.subprocess.Process, grace: float) -> typing.Any:
    """Return the status of a child that ends on its own within `grace`.

    `None` means "still running": the normal case for a long-lived victim,
    which the caller then terminates. A status is only returned once asyncio
    has reaped the child, so the caller can trust it.
    """
    if proc is None:
        return None
    try:
        return await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        return None


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
    early_errors: typing.List[str] = []
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
        await asyncio.sleep(10)
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
            handle_stdout(
                agent.stdout, should_exit, "AGENT",
                [(expected_str_signal, expected_str)], True, early_errors,
            )
        )
        if bashreadline_patch:
            return
        await asyncio.wait_for(expected_str_signal.wait(), AGENT_TIMEOUT)
    finally:
        failed = None
        should_exit.set()
        try:
            out_tasks = [task for task in (server_out, agent_out) if task is not None]
            # The gather below is the only place the children are reaped, so it
            # is also the first point where their status can be read: until it
            # completes, asyncio leaves returncode at None.
            for task, outcome in zip(
                out_tasks,
                await asyncio.gather(*out_tasks, return_exceptions=True),
            ):
                if isinstance(outcome, BaseException) and failed is None:
                    failed = f"{task.get_name()}: {outcome!r}"
            if early_errors and failed is None:
                failed = early_errors[0]
            # A child that ends on its own right after the expected output is
            # the failure #342 describes, and asyncio only fills in
            # `returncode` when it reaps. Wait a bounded moment for that, in
            # parallel so a healthy run pays the grace once, then treat a
            # still-running child as the normal long-lived victim.
            names = (("SERVER", server), ("AGENT", agent))
            for (name, _), code in zip(
                names, await asyncio.gather(*(exited_early(p, EXIT_GRACE) for _, p in names))
            ):
                if code not in (None, 0):
                    if failed is None:
                        failed = f"{name} exited with {code} before the expected output"
                    print(f"{name}: exited with {code}")
            await terminate_process(agent, "AGENT")
            await terminate_process(server, "SERVER")
        except Exception as ex:
            if failed is None:
                failed = repr(ex)
        # `sys.exc_info()` is the exception leaving the `try` above, if any.
        # When one is already unwinding, report the work result but leave the
        # status alone: the propagating exception is the more useful failure.
        propagating = sys.exc_info()[0] is not None
        if failed is not None:
            print(f"FAILED: {failed}")
            if not propagating:
                raise SystemExit(failed)
        elif not propagating:
            print("Test successfully")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
