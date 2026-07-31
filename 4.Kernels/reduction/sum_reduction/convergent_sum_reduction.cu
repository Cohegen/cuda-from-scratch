#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

/*
    Convergent (Sequential Addressing) Sum Reduction

    Improvements over the naive implementation:

    - Reduced control divergence.
    - Threads participating in each iteration are contiguous.
    - Still uses global memory.
    - Operates on a single CUDA block.
    - Modifies the input tensor in-place.

    Assumes:
        input.numel() == 2 * blockDim.x
*/

__global__ void convergent_sum_reduction(
    float* input,
    float* output)
{
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (i < stride)
        {
            input[i] += input[i + stride];
        }

        __syncthreads();
    }

    if (i == 0)
    {
        output[0] = input[0];
    }
}


/*
    PyTorch launcher
*/

torch::Tensor sum(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(),
                "Input must be a CUDA tensor.");

    TORCH_CHECK(input.dtype() == torch::kFloat32,
                "Input must be float32.");

    TORCH_CHECK(input.is_contiguous(),
                "Input must be contiguous.");

    const int threads = input.numel() / 2;

    auto output = torch::zeros({1}, input.options());

    convergent_sum_reduction<<<1, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>());

    cudaDeviceSynchronize();

    return output;
}


/*
    Python bindings
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "sum",
        &sum,
        "Convergent CUDA Sum Reduction");
}
