import pathlib
import sys
import asyncio
import typing
import signal
import requests
CONTROLLER_TIMEOUT = 30
NGINX_TIMEOUT = 30


async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback_all: typing.List[typing.Tuple[asyncio.Event, str]] = [],
    check_error: bool = False,
):
    print("Monitoring output of", title)
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
            if check_error and "[error]" in s:
                assert False, "Error occurred in agent!"
        if t1 in done:
            break
        if stdout.at_eof():
            break


async def main():
    nginx_bin, controller, attach_implementation_dir = sys.argv[1:]
    attach_implementation_dir = pathlib.Path(
        attach_implementation_dir).absolute()
    print("Starting controller at", controller)
    try:
        controller = await asyncio.subprocess.create_subprocess_exec(
            controller,
            "/aaaa",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        print("Waiting for controller to start..")
        should_exit = asyncio.Event()
        controller_started_signal = asyncio.Event()
        controller_handler = asyncio.create_task(
            handle_stdout(
                controller.stdout,
                should_exit,
                "CONTROLLER",
                [
                    (controller_started_signal, "Epoll fd is")
                ]
            )
        )
        await asyncio.wait_for(controller_started_signal.wait(), CONTROLLER_TIMEOUT)
        await asyncio.sleep(3)
        print("Controller started!")

        nginx = await asyncio.create_subprocess_exec(
            nginx_bin,
            "-p",
            str(attach_implementation_dir),
            "-c",
            "./nginx.conf",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=attach_implementation_dir
        )
        nginx_started_signal = asyncio.Event()
        nginx_handler = asyncio.create_task(
            handle_stdout(
                nginx.stdout, should_exit, "NGINX", [
                    (nginx_started_signal, "Module init (0 is success): 0")]
            )
        )
        await asyncio.wait_for(nginx_started_signal.wait(), NGINX_TIMEOUT)
        await asyncio.sleep(3)
        print("nginx started!")
        # Try sending some requests..
        loop = asyncio.get_event_loop()
        session = requests.Session()
        session.trust_env = False
        should_fail_request = await loop.run_in_executor(None, session.get, "http://127.0.0.1:9023/aaab")
        print("Request /aaab")
        print(should_fail_request.text)
        assert should_fail_request.status_code == 403
        should_succeed_request = await loop.run_in_executor(None, session.get, "http://127.0.0.1:9023/aaaaaab")
        
        print("Request /aaaaaabaaab")
        print(should_succeed_request.text)
        assert should_succeed_request.status_code == 200
        print("Test done!")

    finally:
        should_exit.set()
        try:
            controller.send_signal(signal.SIGINT)
            nginx.send_signal(signal.SIGINT)
            await asyncio.gather(controller_handler, nginx_handler)
            # for task in asyncio.all_tasks():
            #     task.cancel()
            await asyncio.gather(controller.communicate(), nginx.communicate())
        except Exception as ex:
            print(ex)
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
