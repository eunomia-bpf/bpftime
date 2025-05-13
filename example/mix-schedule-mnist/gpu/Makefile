gpu_runner: main.cu utils.cpp utils.hpp 
	clang++-17 ./main.cu ./utils.cpp -Wall --cuda-gpu-arch=sm_60 -O2 -L/usr/local/cuda/lib64/ -lcudart -o gpu_runner
