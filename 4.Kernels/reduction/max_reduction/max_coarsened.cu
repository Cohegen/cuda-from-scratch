/*
    Thread-Coarsened Maximum Reduction

    Each thread computes the maximum of
    2 * COARSE_FACTOR elements from global memory,
    stores its local maximum in shared memory,
    then performs a shared-memory reduction.

    Each block produces one partial maximum.
*/

#include <cuda_runtime.h>
#include <algorithm>

constexpr unsigned int BLOCK_DIM = 256;
constexpr unsigned int COARSE_FACTOR = 4;

__global__ void max_coarsened_kernel(
    const float* input,
    float* output,
    unsigned int N
)
{
    __shared__ float input_shared[BLOCK_DIM];

    unsigned int t = threadIdx.x;

    unsigned int segment =
        blockIdx.x * blockDim.x * COARSE_FACTOR * 2;

    unsigned int i = segment + t;

    // Initialize local maximum
    float max_t = -FLT_MAX;

    // Thread coarsening
    for (unsigned int tile = 0; tile < 2 * COARSE_FACTOR; ++tile)
    {
        unsigned int idx = i + tile * BLOCK_DIM;

        if (idx < N)
        {
            max_t = fmaxf(max_t, input[idx]);
        }
    }

    // Store local maximum in shared memory
    //each thread has it's own private copy of max_t
    input_shared[t] = max_t;
    __syncthreads();

    // Shared-memory reduction
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (t < stride)
        {
            input_shared[t] =
                fmaxf(input_shared[t],
                      input_shared[t + stride]);
        }

        __syncthreads();
    }

    // Write block maximum
    if (t == 0)
    {
        output[blockIdx.x] = input_shared[0];
    }
}