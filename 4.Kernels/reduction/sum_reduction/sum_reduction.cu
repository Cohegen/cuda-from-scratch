#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

/*
    Naive parallel sum reduction.

    - Uses only global memory.
    - One block performs the reduction.
    - Input size must be <= 2 * blockDim.x.
    - The input tensor is modified in-place.
*/

__global__ void sum_reduction_kernel(
    float* input,
    float* output)
{
    // Thread i is responsible for element 2*i
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1;
         stride <= blockDim.x;
         stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            input[i] += input[i + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[0] = input[0];
    }
}


/*
    PyTorch launcher

    Assumes:
        input.numel() <= 512
        THREADS = input.numel() / 2
*/

torch::Tensor sum(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor.");
    TORCH_CHECK(input.dtype() == torch::kFloat32,
                "Input must be float32.");
    TORCH_CHECK(input.is_contiguous(),
                "Input must be contiguous.");

    auto output = torch::zeros({1}, input.options());

    const int threads = input.numel() / 2;

    sum_reduction_kernel<<<1, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>());

    cudaDeviceSynchronize();

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "sum",
        &sum,
        "Naive CUDA Sum Reduction");
}
