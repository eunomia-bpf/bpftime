


function(find_cuda)
    if(NOT BPFTIME_CUDA_ROOT)
        message(FATAL_ERROR "To use NV attach, set BPFTIME_CUDA_ROOT to the root of CUDA installation, such as /usr/local/cuda-12.6")
    endif()
    set(CUDA_LIBRARY_PATH ${BPFTIME_CUDA_ROOT}/targets/x86_64-linux/lib/ ${BPFTIME_CUDA_ROOT}/targets/x86_64-linux/lib/stubs/ ${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64/ PARENT_SCOPE)
    set(CUDA_INCLUDE_PATH ${BPFTIME_CUDA_ROOT}/targets/x86_64-linux/include ${BPFTIME_CUDA_ROOT}/cuda-12.6/extras/CUPTI/include PARENT_SCOPE)
    set(CUDA_LIBS cuda cudart libnvptxcompiler_static.a libcupti_static.a nvrtc PARENT_SCOPE)
endfunction()
