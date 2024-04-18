rm -rf ./build/*

if [ $# -eq 2 ]; then
    SM=$1
else
    SM=90
fi

nvcc gemm_test.cu -o ./build/gemm_test -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -std=c++17 -gencode=arch=compute_${SM},code=sm_${SM}

# disassembly
cuobjdump -sass build/gemm_test > fma_bench.sass


