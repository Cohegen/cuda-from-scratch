/*
2D Tiling CUDA Kernel Example
Demonstrates loading 2D matrix tiles into shared memory to compute matrix addition efficiently.
*/

#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA 2D Tiling Kernel Entry Point
__global__ void tileKernel(const float *A, const float *B, float *C, int width, int height)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // Load data into shared memory tiles
    if (row < height && col < width)
    {
        tileA[ty][tx] = A[row * width + col];
        tileB[ty][tx] = B[row * width + col];
    }
    else
    {
        tileA[ty][tx] = 0.0f;
        tileB[ty][tx] = 0.0f;
    }

    // Synchronize threads to guarantee tile loading completes
    __syncthreads();

    // Compute result from shared memory tiles
    if (row < height && col < width)
    {
        C[row * width + col] = tileA[ty][tx] + tileB[ty][tx];
    }
}

// Host launcher function example
void launchTileKernel(const float *d_A, const float *d_B, float *d_C, int width, int height)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    tileKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
}