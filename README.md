# Hopper/Ampere cuda core peak test

## How to build
After cloned code, then cd to the code repo.

mkdir build/

### On Hopper:
sh script/build.sh 90
./build/gemm_test 

and I got FP32 55 tflops and BF16 57 tflops.

### On Ampere:
sh script/build.sh 80
./build/gemm_test

and I got FP32 18 tflops and BF16 35 tflops.
