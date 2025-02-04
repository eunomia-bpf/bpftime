
function(find_cuda)
    set(CUDA_LIBRARY_PATH /usr/local/cuda-12.6/targets/x86_64-linux/lib/ /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/ /usr/local/cuda-12.6/extras/CUPTI/lib64/ PARENT_SCOPE)
    set(CUDA_INCLUDE_PATH /usr/local/cuda-12.6/targets/x86_64-linux/include /usr/local/cuda-12.6/extras/CUPTI/include PARENT_SCOPE)
    set(CUDA_LIBS cuda cudart libnvptxcompiler_static.a libcupti_static.a PARENT_SCOPE)
endfunction()
