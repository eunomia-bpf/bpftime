import asyncio
import pathlib
import typing
from io import StringIO
import argparse

WRK_PATH = "../../assets/wrk"
DEEPFLOW = "../../assets/rust_sample"
BPFTIME_DAEMON = "../../assets/bpftime_daemon"
BPFTIME_CLI = "../../assets/bpftime"

wrk_conn = -1
wrk_thread = -1
no_uprobe = False


def parse_wrk_output(text: str) -> typing.Tuple[float, float]:
    req_sec = -1
    transfer_sec = -1
    for line in text.splitlines():
        if line.startswith("Requests/sec:"):
            req_sec = line.replace("Requests/sec:", "").strip()
        elif line.startswith("Transfer/sec:"):
            transfer_sec = line.replace("Transfer/sec:", "").strip()
    if transfer_sec.endswith("MB"):
        transfer_sec = float(transfer_sec[:-2])
    elif transfer_sec.endswith("GB"):
        transfer_sec = float(transfer_sec[:-2]) * 1024
    return float(req_sec), float(transfer_sec)


async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback: typing.Optional[typing.Tuple[asyncio.Event, str]] = None,
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
                evt, sig = callback
                if sig in s:
                    evt.set()
                    print("Callback triggered")
        if t1 in done:
            break
        if stdout.at_eof():
            break


async def run_wrk(url: str) -> str:
    # Run wrk
    wrk = await asyncio.subprocess.create_subprocess_exec(
        WRK_PATH,
        "-c",
        str(wrk_conn),
        "-t",
        str(wrk_thread),
        url,
        stdout=asyncio.subprocess.PIPE,
    )
    print("WRK started")
    stdout, _ = await wrk.communicate()
    return stdout.decode()


async def run_server(executable: str, data_size: int) -> asyncio.subprocess.Process:
    wd = pathlib.Path(executable).parent
    server = await asyncio.subprocess.create_subprocess_exec(
        executable,
        str(data_size),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=wd,
        env={"BPFTIME_USE_JIT": "1", "GOMAXPROCS": "20"},
    )
    return server


# Run test, without probing
async def run_vanilla_test(executable: str, data_size: int, url: str) -> str:
    try:
        should_exit = asyncio.Event()
        server_start_cb = asyncio.Event()
        server = await run_server(executable, data_size)
        done = asyncio.get_running_loop().create_task(
            handle_stdout(
                server.stdout,
                should_exit,
                "SERVER",
                (server_start_cb, "Server started!"),
            )
        )
        await asyncio.wait_for(server_start_cb.wait(), timeout=10)
        print("Server started")
        await asyncio.sleep(1)
        wrk_out = await run_wrk(url)
        return wrk_out
    finally:
        should_exit.set()
        await done
        server.kill()
        await server.communicate()


async def start_deepflow(
    shutup_after_start: bool = True,
) -> typing.Tuple[asyncio.subprocess.Process, asyncio.Task]:
    # Run deepflow
    env = {}
    if no_uprobe:
        env["NO_UPROBE"] = "1"
    deepflow = await asyncio.subprocess.create_subprocess_exec(
        DEEPFLOW,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    deepflow_start_cb = asyncio.Event()
    shut_up_deepflow = asyncio.Event()
    deepflow_out = asyncio.create_task(
        handle_stdout(
            deepflow.stdout,
            shut_up_deepflow,
            "DEEPFLOW",
            (deepflow_start_cb, "tracer_start() finish"),
        )
    )
    # Wait until deepflow starts..
    await asyncio.wait_for(deepflow_start_cb.wait(), 100)
    # shut that thing up
    if shutup_after_start:
        shut_up_deepflow.set()
    return deepflow, deepflow_out


# Run test, with kernel probing
async def run_kernel_probe_test(
    server_executable: str, data_size: int, url: str
) -> str:
    try:
        server = await run_server(server_executable, data_size)
        should_exit = asyncio.Event()
        server_start_cb = asyncio.Event()
        server_out = asyncio.get_running_loop().create_task(
            handle_stdout(
                server.stdout,
                should_exit,
                "SERVER",
                (server_start_cb, "Server started!"),
            )
        )
        await asyncio.wait_for(server_start_cb.wait(), timeout=10)
        await asyncio.sleep(1)
        print("Server started")
        deepflow, deepflow_out = await start_deepflow()
        print("Deepflow started!")
        wrk_out = await run_wrk(url)
        return wrk_out
    finally:
        should_exit.set()
        server.kill()
        deepflow.kill()
        await asyncio.gather(server_out, deepflow_out)
        await asyncio.gather(server.communicate(), deepflow.communicate())


async def run_userspace_probe_test(
    server_executable: str, data_size: int, url: str
) -> str:
    try:
        server = await run_server(server_executable, data_size)
        should_exit = asyncio.Event()
        server_start_cb = asyncio.Event()
        server_out = asyncio.get_running_loop().create_task(
            handle_stdout(
                server.stdout,
                should_exit,
                "SERVER",
                (server_start_cb, "Server started!"),
            )
        )
        await asyncio.wait_for(server_start_cb.wait(), timeout=10)
        await asyncio.sleep(1)
        print("Server started")
        # Run bpftime-daemon
        bpftime_daemon = await asyncio.subprocess.create_subprocess_exec(
            BPFTIME_DAEMON,
            "-w",
            "0x38e40",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        bpftime_daemon_out = asyncio.get_running_loop().create_task(
            handle_stdout(
                bpftime_daemon.stdout,
                should_exit,
                "DAEMON",
            )
        )
        await asyncio.sleep(3)
        # Run deepflow
        deepflow, deepflow_out = await start_deepflow(True)
        print("Deepflow started!")
        # Attach it
        out_str, _ = await (
            await asyncio.subprocess.create_subprocess_shell(
                f"{BPFTIME_CLI} attach {server.pid}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        ).communicate()
        print("BPFTIME_CLI:", out_str.decode(), end="")
        await asyncio.sleep(1)
        # pre-fetch
        out, _ = await (
            await asyncio.subprocess.create_subprocess_shell(
                f"wget --no-proxy --no-check-certificate -O /dev/null {url}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        ).communicate()
        print("WGET:", out.decode(), end="")
        # Attached!
        wrk_out = await run_wrk(url)
        return wrk_out
    finally:
        should_exit.set()
        server.kill()
        deepflow.kill()
        bpftime_daemon.kill()
        await asyncio.gather(server_out, deepflow_out, bpftime_daemon_out)
        await asyncio.gather(
            server.communicate(), deepflow.communicate(), bpftime_daemon.communicate()
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="Test type (no-probe, kernel-uprobe, no-uprobe, user-uprobe)",
        action="store",
        choices=["no-probe", "kernel-uprobe", "no-uprobe", "user-uprobe"],
        required=True,
    )
    parser.add_argument("--http", help="Test with http server", action="store_true")
    parser.add_argument("--https", help="Test with https server", action="store_true")
    parser.add_argument(
        "--wrk-thread", help="wrk thread count", action="store", type=int, default=10
    )
    parser.add_argument(
        "--wrk-conn",
        help="wrk connection count",
        action="store",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    global wrk_conn, wrk_thread
    wrk_conn = args.wrk_conn
    wrk_thread = args.wrk_thread
    if (args.http ^ args.https) == 0:
        print("You CAN NOT set or unset http&https at the sametime")
        exit(1)
    run_func = None
    if args.type == "no-probe":
        run_func = run_vanilla_test
    elif args.type == "kernel-uprobe":
        run_func = run_kernel_probe_test
    elif args.type == "no-uprobe":
        global no_uprobe
        no_uprobe = True
        run_func = run_kernel_probe_test
    elif args.type == "user-uprobe":
        run_func = run_userspace_probe_test
    else:
        assert False
    use_https = args.https
    DATA_SIZES = [x * 1024 for x in [1, 2, 4, 16, 128, 256]]
    out = []
    for size in DATA_SIZES:
        req, trans = 0, 0
        N = 1
        for _ in range(N):
            print(f"Running test for size {size}")
            result = await run_func(
                "../go-server/main" if use_https else "../go-server-http/main",
                size,
                "https://127.0.0.1:446" if use_https else "http://127.0.0.1:447",
            )
            print(result)
            a, b = parse_wrk_output(result)
            req += a
            trans += b
        req /= N
        trans /= N
        out.append((req, trans))
    print(out)
    print("run_func", run_func)
    print("type", args.type),
    print("use http", args.http)
    print("use https", args.https)
    print("wrk conn", wrk_conn)
    print("wrk thd", wrk_thread)
    buf = StringIO()
    buf.write("| Data Size | Requests/sec | Transfer/sec |\n")
    buf.write("|-----------|--------------|--------------|\n")

    def pad(s: str, n: int = 14) -> str:
        if len(s) < n:
            return s + " " * (n - len(s))
        else:
            return s

    for size, v in zip(DATA_SIZES, out):
        req, trans = v
        buf.write(
            f"|{pad((str(size//1024)+' KB') if size>=1024 else (str(size)+' B'),11)}|{pad(str(req),14)}|{pad(str(trans)+'MB',14)}|\n"
        )
    print(buf.getvalue())


if __name__ == "__main__":
    asyncio.run(main())
