/*
Kernel that adds two matrices
*/

#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void MatAddKernelCUDA(float *M, float *N, float *P, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < width && col < width)
    {
        int idx = row * width + col;
        P[idx] = M[idx] + N[idx];
    }
}

torch::Tensor MatAddCUDA(torch::Tensor M, torch::Tensor N)
{
    TORCH_CHECK(M.device().is_cuda(), "M must be CUDA tensor");
    TORCH_CHECK(N.device().is_cuda(), "N must be CUDA tensor");

    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "M must be float32");
    TORCH_CHECK(N.scalar_type() == torch::kFloat32, "N must be float32");

    TORCH_CHECK(M.dim() == 2 && N.dim() == 2, "M and N must be 2D tensors");

    TORCH_CHECK(M.is_contiguous() && N.is_contiguous(), "Tensors must be contiguous");

    TORCH_CHECK(M.sizes() == N.sizes(), "M and N must have same shape");

    int width = M.size(0);

    auto P = torch::empty_like(M);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + 15) / 16,
        (width + 15) / 16
    );

    MatAddKernelCUDA<<<blocksPerGrid, threadsPerBlock, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(),
        N.data_ptr<float>(),
        P.data_ptr<float>(),
        width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return P;
}
