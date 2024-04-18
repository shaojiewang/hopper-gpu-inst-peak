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
};
template<>
struct a_vec_type<__nv_bfloat16, 1>
{
    using type = __nv_bfloat16;
};
template<>
struct a_vec_type<__nv_bfloat16, 2>
{
    using type = __nv_bfloat162;
};

template<class T>
struct vector_num
{
    static constexpr int vec = sizeof(float) / sizeof(T);
};

template<class T>
__global__ void fma_block(T* a, T* b, T* c)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx * blockDim.x + tidx;

    constexpr int vec = vector_num<T>::vec;
    using vec_type = typename a_vec_type<T, vec>::type;
    vec_type a_vec; // = reinterpret_cast<vec_type*>(a)[offset];
    vec_type b_vec; // = reinterpret_cast<vec_type*>(b)[offset];
    vec_type c_vec;

    if constexpr(std::is_same<vec_type, float>::value)
    {
        a_vec = 5.0f;
        b_vec = 1.0f;
    }
    else if constexpr(std::is_same<vec_type, __nv_bfloat162>::value)
    {
        a_vec = __float22bfloat162_rn(make_float2(5.0f, 5.0f));
        b_vec = __float22bfloat162_rn(make_float2(1.0f, 1.0f));
    }
    
    uint32_t* c_ = reinterpret_cast<uint32_t*>(&c_vec);
    uint32_t a_ = 1;//*(reinterpret_cast<uint32_t*>(&a_vec));
    uint32_t b_ = 2;//*(reinterpret_cast<uint32_t*>(&b_vec));
    uint32_t d_ = 3;//*(reinterpret_cast<uint32_t*>(&b_vec));

    uint16_t a_uint16 = 1;
    uint16_t b_uint16 = 2;
    uint16_t* c_uint16 = reinterpret_cast<uint16_t*>(&c_vec);


#pragma unroll
    for(int i = 0; i < LOOP_NUM; i++)
    {
        if constexpr(std::is_same<vec_type, float>::value)
        {
            // c_vec += a_vec + b_vec;
            c_vec = fma(a_vec, b_vec, c_vec);
        }
        else if constexpr(std::is_same<vec_type, __nv_bfloat162>::value)
        {
            asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(*c_) : "r"(a_), "r"(b_), "r"(*c_));
            // asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=r"(*c_) : "r"(a_), "r"(b_), "r"(*c_));
            // asm volatile("fma.rn.f16 %0, %1, %2, %3;\n" : "=h"(*c_uint16) : "h"(a_uint16), "h"(b_uint16), "h"(*c_uint16));
        }
    }

    reinterpret_cast<vec_type*>(c)[offset] = c_vec;
    
}

template <class T>
void invoke_fma_block(T* a, T* b, T* c, int grid_size, cudaStream_t& stream)
{
    constexpr int cta_size = CUDA_CTA_SIZE;
    int grid = grid_size;
    fma_block<T><<<grid, cta_size, 0, stream>>>(a, b, c);
}

