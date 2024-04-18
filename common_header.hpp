#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define GPU_CHECK_ERROR(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("[cuda error](%d), failed to call %s \n", (int)err, #call); \
        exit(0); \
    } \
} while(0) \

#define TIMEIT(print, n, ms, stream, fn, ...)                                                                              \
    ({                                                                                                                 \
        cudaEvent_t _macro_event_start, _macro_event_stop;                                                             \
        cudaEventCreate(&_macro_event_start);                                                                          \
        cudaEventCreate(&_macro_event_stop);                                                                           \
        cudaEventRecord(_macro_event_start, stream);                                                                   \
        for (int i = 0; i < n; i++) {                                                                                  \
            fn(__VA_ARGS__);                                                                                           \
        }                                                                                                              \
        cudaEventRecord(_macro_event_stop, stream);                                                                    \
        cudaStreamSynchronize(stream);                                                                                 \
        cudaEventElapsedTime(&ms, _macro_event_start, _macro_event_stop);                                              \
        ms /= n;                                                                                                       \
        if (print)                                                                                                     \
            printf("[TIMEIT] " #fn ": %.2fµs\n", ms * 1000);                                                           \
        ms;                                                                                                            \
    })

