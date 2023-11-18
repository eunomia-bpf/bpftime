wrk/wrk https://127.0.0.1:4043/index.html -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example1k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example2k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example4k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example16k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example128k.txt -c 512 -t 4 -d 10 >> test-log.txt
wrk/wrk https://127.0.0.1:4043/data/example256k.txt -c 512 -t 4 -d 10 >> test-log.txt
