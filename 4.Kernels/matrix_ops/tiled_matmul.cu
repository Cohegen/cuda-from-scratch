#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_WIDTH 16

__global__ void matmulKernel(float *M, float *N, float *P, int width)
{
    // defining the tiles stored in shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // automatic variables stored in registers
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // identifying the row and column of the element in P to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph)
    {
        // loading M tile into shared memory
        Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];

        // loading N tile into shared memory
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];

        // employing thread synchronization
        __syncthreads();

        // compute partial product
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    // write final result
    P[Row * width + Col] = Pvalue;
}

torch::Tensor tiled_matmul(torch::Tensor M, torch::Tensor N)
{
    TORCH_CHECK(M.device().is_cuda(), "M must be a CUDA tensor");
    TORCH_CHECK(N.device().is_cuda(), "N must be a CUDA tensor");
    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "M must be float32");
    TORCH_CHECK(N.scalar_type() == torch::kFloat32, "N must be float32");
    TORCH_CHECK(M.dim() == 2 && N.dim() == 2, "M and N must be 2D tensors");
    TORCH_CHECK(M.is_contiguous() && N.is_contiguous(), "M and N must be contiguous");
    TORCH_CHECK(M.size(0) == M.size(1), "M must be square");
    TORCH_CHECK(N.size(0) == N.size(1), "N must be square");
    TORCH_CHECK(M.size(0) == N.size(0), "M and N must have the same width");
    TORCH_CHECK(M.size(0) % TILE_WIDTH == 0,
        "tiled_matmul requires width to be divisible by TILE_WIDTH");

    const int width = M.size(0);
    auto P = torch::empty_like(M);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(width / TILE_WIDTH, width / TILE_WIDTH);

    matmulKernel<<<blocksPerGrid, threadsPerBlock, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(),
        N.data_ptr<float>(),
        P.data_ptr<float>(),
        width);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return P;
}

