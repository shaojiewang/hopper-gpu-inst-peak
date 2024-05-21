#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

#include "common_header.hpp"

#define LOOP_NUM 1000000
#define CUDA_CTA_SIZE 1024

template<class T, int N>
struct a_vec_type
{
};

template<>
struct a_vec_type<float, 1>
{
    using type = float;
    __device__ __forceinline__ static type init(float v) {
        return v;
    }
    __device__ __forceinline__ static void fma(type& va, type& vb, type& vc) {
        asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(vc) : "f"(va), "f"(vb), "f"(vc));
    }
};

template<>
struct a_vec_type<__nv_bfloat16, 2>
{
    using type = __nv_bfloat162;
    __device__ __forceinline__ static type init(float v) {
        return __float2bfloat162_rn(v);
    }
    __device__ __forceinline__ static void fma(type& va, type& vb, type& vc) {
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : 
                     "=r"(reinterpret_cast<uint32_t&>(vc)) : 
                     "r"(reinterpret_cast<uint32_t&>(va)), 
                     "r"(reinterpret_cast<uint32_t&>(vb)), 
                     "r"(reinterpret_cast<uint32_t&>(vc)));
    }
};

template<>
struct a_vec_type<half, 2>
{
    using type = half2;
    __device__ __forceinline__ static type init(float v) {
        return __float2half2_rn(v);
    }
    __device__ __forceinline__ static void fma(type& va, type& vb, type& vc) {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : 
                     "=r"(reinterpret_cast<uint32_t&>(vc)) : 
                     "r"(reinterpret_cast<uint32_t&>(va)), 
                     "r"(reinterpret_cast<uint32_t&>(vb)), 
                     "r"(reinterpret_cast<uint32_t&>(vc)));
    }
};

template<class T>
struct vector_num
{
    static constexpr int vec = sizeof(float) / sizeof(T);
};

#define PARALLEL 16

template<class T>
__global__ void fma_block(T* c)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = (bidx * blockDim.x + tidx) * PARALLEL;

    constexpr int vec = vector_num<T>::vec;
    using vec_type = typename a_vec_type<T, vec>::type;
    vec_type a_vec = reinterpret_cast<vec_type*>(c)[0];
    vec_type b_vec = reinterpret_cast<vec_type*>(c)[1];
    vec_type c_vec[PARALLEL];
    for(int i = 0; i < PARALLEL; ++i) {
        c_vec[i] = a_vec_type<T, vec>::init(0.f);
    }

#pragma unroll
    for(int n = 0; n < LOOP_NUM / PARALLEL; n++)
    {
#pragma unroll
        for(int i = 0; i < PARALLEL; ++i) {
            a_vec_type<T, vec>::fma(a_vec, b_vec, c_vec[i]);
        }
    }
#pragma unroll
    for(int i = 0; i < PARALLEL; ++i) {
        reinterpret_cast<vec_type*>(c)[offset + i] = c_vec[i];
    }    
}

template <class T>
void invoke_fma_block(T* c, int grid_size, cudaStream_t& stream)
{
    constexpr int cta_size = CUDA_CTA_SIZE;
    int grid = grid_size;
    fma_block<T><<<grid, cta_size, 0, stream>>>(c);
}

