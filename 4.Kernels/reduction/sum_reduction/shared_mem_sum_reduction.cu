/*
    Shared Memory Sum Reduction

    Each block reduces 2 * BLOCK_DIM elements into a single value.

    Assumes:
    - BLOCK_DIM is known at compile time.
    - gridDim.x * (2 * BLOCK_DIM) >= number of input elements.
*/

#include <cuda_runtime.h>

constexpr unsigned int BLOCK_DIM = 256;

__global__ void shared_memory_sum_reduction(const float* input,
                                            float* output)
{
    // Shared memory buffer
    __shared__ float input_shared[BLOCK_DIM];

    unsigned int t = threadIdx.x;
    unsigned int global_index = blockIdx.x * (2 * blockDim.x) + t;

    // Each thread loads two values from global memory
    input_shared[t] =
        input[global_index] +
        input[global_index + blockDim.x];

    __syncthreads();

    // Tree reduction
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

    // Write one result per block
    if (t == 0)
    {
        output[blockIdx.x] = input_shared[0];
    }
}

//torch launcher
torch::Tensor shared_memory_sum_reduction_launcher(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(),
                "Input must be a CUDA tensor.");

    TORCH_CHECK(input.dtype() == torch::kFloat32,
                "Input must be float32.");

    TORCH_CHECK(input.is_contiguous(),
                "Input must be contiguous.");

    TORCH_CHECK(
        input.numel() % (2 * BLOCK_DIM) == 0,
        "Input size must be divisible by 2 * BLOCK_DIM.");

    const int num_elements = input.numel();

    const int num_blocks =
        num_elements / (2 * BLOCK_DIM);

    auto output = torch::zeros(
        {num_blocks},
        input.options());

    shared_memory_sum_reduction<<<
        num_blocks,
        BLOCK_DIM,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}