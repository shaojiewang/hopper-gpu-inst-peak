rm -rf ./build/*

nvcc gemm_test.cu -o ./build/gemm_test -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -std=c++17 -gencode=arch=compute_90,code=sm_90

