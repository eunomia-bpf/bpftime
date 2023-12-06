import asyncio
import typing
import argparse
import os
import time
import signal
import json


async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback: typing.Optional[
        typing.Tuple[asyncio.Event, str, typing.List[str]]
    ] = None,
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
            if callback:
                evt, sig, ret = callback
                if sig in s:
                    evt.set()
                    ret.append(s)
                    print("Callback triggered")
        if t1 in done:
            break
        if stdout.at_eof():
            break


async def run_server(
    should_exit: asyncio.Event,
) -> typing.Tuple[asyncio.subprocess.Process, asyncio.Task]:
    server = await asyncio.create_subprocess_exec(
        args.SERVER,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={"LD_PRELOAD": args.server},
    )
    server_start_signal = asyncio.Event()
    server_stdout = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (
                server_start_signal,
                "Created uprobe/uretprobe perf event handler",
                [],
            ),
        )
    )
    await server_start_signal.wait()
    await asyncio.sleep(1)
    print("Server started!")
    return server, server_stdout


async def handle_client_output(
    client: asyncio.subprocess.Process, should_exit: asyncio.Event
) -> typing.Tuple[int, asyncio.StreamReader]:
    client_start_signal = asyncio.Event()
    client_cb = []
    client_stdout = asyncio.get_running_loop().create_task(
        handle_stdout(
            client.stdout,
            should_exit,
            "CLIENT",
            (
                client_start_signal,
                "STARTED:",
                client_cb,
            ),
        )
    )
    await client_start_signal.wait()
    print("Client started!")
    done_time = int(client_cb[0].strip().split()[-1])
    return done_time, client_stdout


async def run_ld_preload_test() -> int:
    try:
        should_exit = asyncio.Event()
        server, server_stdout = await run_server(should_exit)
        start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        client = await asyncio.create_subprocess_exec(
            args.CLIENT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"LD_PRELOAD": args.agent},
        )
        done_time, client_stdout = await handle_client_output(client, should_exit)
        return done_time - start_time
    finally:
        should_exit.set()
        await asyncio.gather(server_stdout, client_stdout)
        server.kill()
        client.send_signal(signal.SIGINT)
        await asyncio.gather(server.wait(), client.wait())


async def run_client_ld_preload_test() -> int:
    try:
        should_exit = asyncio.Event()
        server, server_stdout = await run_server(should_exit)
        start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        client = await asyncio.create_subprocess_shell(
            f"{args.bpftime} start {args.CLIENT}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"LD_PRELOAD": args.agent},
        )
        done_time, client_stdout = await handle_client_output(client, should_exit)
        return done_time - start_time
    finally:
        should_exit.set()
        await asyncio.gather(server_stdout, client_stdout)
        server.kill()
        client.send_signal(signal.SIGINT)
        await asyncio.gather(server.wait(), client.wait())


async def run_ptrace_attach_test() -> int:
    try:
        should_exit = asyncio.Event()
        server, server_stdout = await run_server(should_exit)

        client = await asyncio.create_subprocess_exec(
            args.CLIENT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        injector = await asyncio.create_subprocess_shell(
            f"{args.bpftime} attach {client.pid}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        injector_stdout = asyncio.get_running_loop().create_task(
            handle_stdout(injector.stdout, should_exit, "INJECTOR", None)
        )
        done_time, client_stdout = await handle_client_output(client, should_exit)
        return done_time - start_time
    finally:
        should_exit.set()
        await asyncio.gather(server_stdout, client_stdout, injector_stdout)
        server.kill()
        client.send_signal(signal.SIGINT)
        await asyncio.gather(server.wait(), client.wait(), injector.wait())


def handle_sequence(seq: typing.List[int]) -> dict:
    avg = sum(seq) / len(seq)
    sqr_sum = sum(x**2 for x in seq) / len(seq)

    result = {
        "max": max(seq),
        "min": min(seq),
        "avg": avg,
        "sqr_dev": sqr_sum - avg**2,
        "std_dev": (sqr_sum - avg**2) ** 0.5,
    }
    return {"raw": seq, "statistics": result}


async def main():
    home = os.environ["HOME"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "SERVER",
        help="Path to the executable that server will be injected",
        action="store",
    )
    parser.add_argument(
        "CLIENT",
        help="Path to the executable that the client will be injected",
        action="store",
    )
    parser.add_argument(
        "-a",
        "--agent",
        help="Path to libbpftime-agent.so",
        action="store",
        default=f"{home}/.bpftime/libbpftime-agent.so",
    )
    parser.add_argument(
        "-s",
        "--server",
        help="Path to libbpftime-syscall-server.so",
        action="store",
        default=f"{home}/.bpftime/libbpftime-syscall-server.so",
    )
    parser.add_argument(
        "-b", "--bpftime", help="Path to bpftime cli", action="store", default="bpftime"
    )
    parser.add_argument(
        "-c", "--count", help="Times to test", action="store", type=int, default=100
    )
    global args
    args = parser.parse_args()
    print(await run_ptrace_attach_test())
    result = {}
    for func, key in zip(
        [
            run_ld_preload_test,
            run_client_ld_preload_test,
            run_ptrace_attach_test,
        ],
        ["ld_preload", "ld_preload_with_cli", "ptrace_with_cli"],
    ):
        result[key] = []
        for i in range(1, args.count + 1):
            print(f"{func} -> {i}")
            result[key].append(await func())
        result[key] = handle_sequence(result[key])
    with open("benchmark-result.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    asyncio.run(main())
