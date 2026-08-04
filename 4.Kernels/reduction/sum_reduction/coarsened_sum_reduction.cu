/*
    Coarsened Sum Reduction

    Each thread loads multiple values from global memory,
    accumulates them into a private register, then performs
    a shared-memory reduction.
*/

#include <cuda_runtime.h>

constexpr unsigned int BLOCK_DIM = 256;
constexpr unsigned int COARSE_FACTOR = 4;

__global__ void CoarsenedSumReductionKernel(
    const float* input,
    float* output)
{
    __shared__ float input_shared[BLOCK_DIM];

    unsigned int t = threadIdx.x;

    // Beginning of this block's segment
    unsigned int segment =
        COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;

    // Global index for this thread
    unsigned int i = segment + t;

    // Private accumulator stored in a register
    float sum = 0.0f;

    // Load multiple values from global memory
    for (unsigned int tile = 0;
         tile < COARSE_FACTOR * 2;
         ++tile)
    {
        sum += input[i + tile * blockDim.x];
    }

    // Store partial sum in shared memory
    input_shared[t] = sum;

    __syncthreads();

    // Shared-memory reduction
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (t < stride)
        {
            input_shared[t] += input_shared[t + stride];
        }

        __syncthreads();
    }

    // Accumulate block result
    if (t == 0)
    {
        atomicAdd(output, input_shared[0]);
    }
}