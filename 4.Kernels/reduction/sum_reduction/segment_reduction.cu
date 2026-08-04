/*
    Segmented Sum Reduction

    Each block reduces a contiguous segment of
    2 * BLOCK_DIM input elements into a single sum.

    The partial sums from every block are accumulated
    into a single output value using atomicAdd().
*/

#include <cuda_runtime.h>

constexpr unsigned int BLOCK_DIM = 256;

__global__ void SegmentedSumReduction(
    const float* input,
    float* output)
{
    // Shared memory for one block
    __shared__ float input_shared[BLOCK_DIM];

    unsigned int t = threadIdx.x;

    // Beginning of this block's segment
    unsigned int segment = 2 * blockDim.x * blockIdx.x;

    // Global index handled by this thread
    unsigned int i = segment + t;

    // Load two elements per thread
    input_shared[t] =
        input[i] +
        input[i + blockDim.x];

    __syncthreads();

    // Parallel reduction in shared memory
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

    // Atomically accumulate the block's sum
    if (t == 0)
    {
        atomicAdd(output, input_shared[0]);
    }
}