
all: cpu_runner

clean:
	rm -rf ./cpu_runner

cpu_runner: cpu_runner.cpp cpu_pytorch.py
	g++ -Wall -o cpu_runner cpu_runner.cpp $(shell python3-config --cflags --ldflags --includes --libs --embed)
