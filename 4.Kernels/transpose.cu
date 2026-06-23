/*
Kernel which performs matrix transposition
*/

#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matTransposeKernel(const float *M, float *N, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < width && col < width)
    {
        int input_idx = row * width + col;
        int output_idx = col * width + row;
        N[output_idx] = M[input_idx];
    }
}

torch::Tensor MatTransposeCUDA(torch::Tensor M)
{
    TORCH_CHECK(M.device().is_cuda(), "M must be CUDA tensor");
    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "M must be float32");
    TORCH_CHECK(M.dim() == 2, "M must be 2D tensor");
    TORCH_CHECK(M.is_contiguous(), "M must be contiguous");

    int width = M.size(0);
    TORCH_CHECK(M.size(1) == width, "Only square matrices supported in this version");

    auto N = torch::empty_like(M);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + 15) / 16,
        (width + 15) / 16
    );

    matTransposeKernel<<<blocksPerGrid, threadsPerBlock, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(),
        N.data_ptr<float>(),
        width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return N;
}
