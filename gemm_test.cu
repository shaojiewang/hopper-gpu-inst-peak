#include <cstdio>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_fma_bench.hpp"

template<class T>
void gemm_test()
{
    T* C;
    int device_id;
    int num_multi_processors;
    constexpr int cta_size = CUDA_CTA_SIZE;
    GPU_CHECK_ERROR(cudaGetDevice(&device_id));
    GPU_CHECK_ERROR(cudaDeviceGetAttribute(&num_multi_processors, cudaDevAttrMultiProcessorCount, device_id));

    num_multi_processors *= 4;
    GPU_CHECK_ERROR(cudaMalloc(&C, PARALLEL * 4 * cta_size * num_multi_processors));
    

    // time it
    cudaStream_t c_stream;
    GPU_CHECK_ERROR(cudaStreamCreate(&c_stream));

    printf("num_multi_processors=%d\n", num_multi_processors);

    // warm up
    invoke_fma_block<T>(C, num_multi_processors, c_stream);

    float ms = 0.0f;
    if constexpr(std::is_same<T, float>::value)
    {
        printf("[fp32] ");
    }
    else if constexpr(std::is_same<T, __nv_bfloat16>::value)
    {
        printf("[bf16] ");
    }
    else if constexpr(std::is_same<T, half>::value)
    {
        printf("[fp16] ");
    }
    TIMEIT(true, 10, ms, c_stream, invoke_fma_block, C, num_multi_processors, c_stream);
    float tflop = (float)num_multi_processors * 2 * LOOP_NUM * CUDA_CTA_SIZE * (sizeof(float) / sizeof(T));
    float tflops = tflop / 1024.0 / 1024.0 / (ms * 1000);
    printf("tflops=%f\n", tflops);
}


int main(int argc, char* argv[])
{
    if(argc < 1)
    {
        printf("gg, you need to pass some args\n");
    }

    gemm_test<float>();
    gemm_test<__nv_bfloat16>();
    gemm_test<half>();

    printf("hello cuda\n");
}

