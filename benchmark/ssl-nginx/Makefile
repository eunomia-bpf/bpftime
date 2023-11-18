index.html: index.html
wrk/wrk:
	git clone https://github.com/wg/wrk
	cd wrk && make

.PHONY: test 
test: test-log-1.txt test-log-2.txt test-log-3.txt 

test-log-1.txt: wrk/wrk
	./test.sh
	mv test-log.txt test-log-1.txt

test-log-2.txt: wrk/wrk
	./test.sh
	mv test-log.txt test-log-2.txt

test-log-3.txt: wrk/wrk
	./test.sh
	mv test-log.txt test-log-3.txt

