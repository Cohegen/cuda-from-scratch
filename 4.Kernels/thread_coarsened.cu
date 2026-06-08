#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4
#define WIDTH 128

__global__ void matmulKernel(float *M, float *N, float *P, int width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c)
    {
        Pvalue[c] = 0.0f;
    }

    int numPhases = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < numPhases; ++ph)
    {
        if (row < width && (ph * TILE_WIDTH + tx) < width)
        {
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; ++c)
        {
            int col = colStart + c * TILE_WIDTH;

            if ((ph * TILE_WIDTH + ty) < width && col < width)
            {
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            }
            else
            {
                Nds[ty][tx] = 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k)
            {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }

            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c)
    {
        int col = colStart + c * TILE_WIDTH;
        if (row < width && col < width)
        {
            P[row * width + col] = Pvalue[c];
        }
    }
}

torch::Tensor thread_coarsened_matmul(torch::Tensor M, torch::Tensor N)
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

    const int width = M.size(0);
    auto P = torch::empty_like(M);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (width + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR),
        (width + TILE_WIDTH - 1) / TILE_WIDTH);

    matmulKernel<<<blocksPerGrid, threadsPerBlock, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(),
        N.data_ptr<float>(),
        P.data_ptr<float>(),
        width);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return P;
}

