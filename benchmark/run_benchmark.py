import asyncio
import typing
import pathlib
import math
import json

CURR_FILE_DIR = pathlib.Path(__file__).parent


def parse_victim_output(v: str) -> dict:
    lines = v.strip().splitlines()
    i = 0
    result = {}
    while i < len(lines):
        curr_line = lines[i]
        if curr_line.startswith("Benchmarking"):
            name = curr_line.split()[1]
            time_usage = float((lines[i + 1]).split()[3])
            result[name] = time_usage
            i += 2

        else:
            i += 1
    return result


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


def handle_result(result: dict) -> dict:
    new_result = {}
    for k, v in result.items():
        avg = sum(v) / len(v)
        sqr_sum = sum(x**2 for x in v)
        sqr_diff = sqr_sum / len(v) - (avg**2)
        val = {"max": max(v), "min": min(v), "avg": avg, "sqr": sqr_diff}
        new_result[k] = val
    return new_result


async def run_userspace_uprobe_test():
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    server = await asyncio.subprocess.create_subprocess_exec(
        str(CURR_FILE_DIR / "uprobe" / "uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=CURR_FILE_DIR.parent,
        env={
            "LD_PRELOAD": str(
                CURR_FILE_DIR.parent
                / "build/runtime/syscall-server/libbpftime-syscall-server.so"
            )
        },
    )
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "__bench_probe is for uprobe only"),
        )
    )
    await server_start_cb.wait()
    print("server started!")
    result = None
    for i in range(10):
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(CURR_FILE_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=CURR_FILE_DIR.parent,
            env={
                "LD_PRELOAD": str(
                    CURR_FILE_DIR.parent / "build/runtime/agent/libbpftime-agent.so"
                )
            },
        )
        victim_out, _ = await victim.communicate()
        curr = parse_victim_output(victim_out.decode())
        if result is None:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                result[k].append(v)
        print(f"{i}: {curr}")
    should_exit.set()
    await server_stdout_task
    server.kill()
    await server.communicate()
    return handle_result(result)


async def run_kernel_uprobe_test():
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    server = await asyncio.subprocess.create_subprocess_exec(
        str(CURR_FILE_DIR / "uprobe" / "uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=CURR_FILE_DIR.parent,
    )
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "__bench_probe is for uprobe only"),
        )
    )
    await server_start_cb.wait()
    print("server started!")
    # await asyncio.sleep(10)
    result = None
    for i in range(10):
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(CURR_FILE_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=CURR_FILE_DIR.parent,
        )
        victim_out, _ = await victim.communicate()
        curr = parse_victim_output(victim_out.decode())
        if result is None:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                result[k].append(v)
        print(f"{i}: {curr}")
    should_exit.set()
    await server_stdout_task
    server.kill()
    await server.communicate()
    return handle_result(result)


def handle_embed_victim_out(i: str) -> float:
    for line in i.splitlines():
        if line.startswith("avg function elapse time:"):
            return float(line.split()[-2])
    return math.inf


async def run_embed_vm_test():
    result = {"embed": []}
    bpf_path = str(CURR_FILE_DIR / "uprobe/.output/uprobe.bpf.o")
    for i in range(10):
        victim = await asyncio.subprocess.create_subprocess_shell(
            " ".join(
                [
                    str(
                        CURR_FILE_DIR.parent
                        / "build/benchmark/simple-benchmark-with-embed-ebpf-calling"
                    ),
                    bpf_path,
                    bpf_path,
                ]
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=CURR_FILE_DIR.parent,
        )
        victim_out, _ = await victim.communicate()
        result["embed"].append(handle_embed_victim_out(victim_out.decode()))
    return handle_result(result)


async def main():
    k = await run_kernel_uprobe_test()
    u = await run_userspace_uprobe_test()
    e = await run_embed_vm_test()
    out = {"kernel_uprobe": k, "userspace_uprobe": u, "embed": e}
    with open("benchmark-output.json", "w") as f:
        json.dump(out, f)
    print(out)

if __name__ == "__main__":
    asyncio.run(main())
