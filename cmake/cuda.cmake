
function(find_cuda)
    set(CUDA_LIBRARY_PATH /usr/local/cuda-12.6/targets/x86_64-linux/lib/ /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/ PARENT_SCOPE)
    set(CUDA_INCLUDE_PATH /usr/local/cuda-12.6/targets/x86_64-linux/include PARENT_SCOPE)
    set(CUDA_LIBS cuda cudart PARENT_SCOPE)
endfunction()
