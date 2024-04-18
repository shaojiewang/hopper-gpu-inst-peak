# Hopper/Ampere cuda core peak test

## How to build
After cloned code, then cd to the code repo.

``` shell
mkdir build/
```

### On Hopper:
``` shell
sh script/build.sh 90

./build/gemm_test 
```

and I got FP32 55 tflops and BF16 57 tflops.

### On Ampere:
``` shell
sh script/build.sh 80

./build/gemm_test
```

and I got FP32 18 tflops and BF16 35 tflops.

A disassemble code could be find in fma_bench.sass
