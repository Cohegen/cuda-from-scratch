/*
    Tiled 2D 5-point stencil using shared memory.

    Each block computes a TILE_WIDTH x TILE_WIDTH tile.
    A one-cell halo is loaded around the tile so that
    neighboring values can be read from shared memory
    instead of global memory.
*/

#include <cuda_runtime.h>
#define TILE_WIDTH 16

__global__ void stencil_tiled_2d(const float* input,
                                 float* output,
                                 int width)
{
    // Shared memory tile including halo cells
    __shared__ float input_s[TILE_WIDTH + 2][TILE_WIDTH + 2];

    // Global coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Local coordinates inside shared memory
    int ty = threadIdx.y + 1;
    int tx = threadIdx.x + 1;

    //----------------------------------------------------
    // Load center element
    //----------------------------------------------------

    if (row < width && col < width)
    {
        input_s[ty][tx] = input[row * width + col];
    }

    //----------------------------------------------------
    // Load left halo
    //----------------------------------------------------

    if (threadIdx.x == 0 &&
        col > 0 &&
        row < width)
    {
        input_s[ty][0] =
            input[row * width + col - 1];
    }

    //----------------------------------------------------
    // Load right halo
    //----------------------------------------------------

    if (threadIdx.x == TILE_WIDTH - 1 &&
        col < width - 1 &&
        row < width)
    {
        input_s[ty][TILE_WIDTH + 1] =
            input[row * width + col + 1];
    }

    //----------------------------------------------------
    // Load top halo
    //----------------------------------------------------

    if (threadIdx.y == 0 &&
        row > 0 &&
        col < width)
    {
        input_s[0][tx] =
            input[(row - 1) * width + col];
    }

    //----------------------------------------------------
    // Load bottom halo
    //----------------------------------------------------

    if (threadIdx.y == TILE_WIDTH - 1 &&
        row < width - 1 &&
        col < width)
    {
        input_s[TILE_WIDTH + 1][tx] =
            input[(row + 1) * width + col];
    }

    //----------------------------------------------------
    // Load corner halos
    //----------------------------------------------------

    // Top-left
    if (threadIdx.x == 0 &&
        threadIdx.y == 0 &&
        row > 0 &&
        col > 0)
    {
        input_s[0][0] =
            input[(row - 1) * width + col - 1];
    }

    // Top-right
    if (threadIdx.x == TILE_WIDTH - 1 &&
        threadIdx.y == 0 &&
        row > 0 &&
        col < width - 1)
    {
        input_s[0][TILE_WIDTH + 1] =
            input[(row - 1) * width + col + 1];
    }

    // Bottom-left
    if (threadIdx.x == 0 &&
        threadIdx.y == TILE_WIDTH - 1 &&
        row < width - 1 &&
        col > 0)
    {
        input_s[TILE_WIDTH + 1][0] =
            input[(row + 1) * width + col - 1];
    }

    // Bottom-right
    if (threadIdx.x == TILE_WIDTH - 1 &&
        threadIdx.y == TILE_WIDTH - 1 &&
        row < width - 1 &&
        col < width - 1)
    {
        input_s[TILE_WIDTH + 1][TILE_WIDTH + 1] =
            input[(row + 1) * width + col + 1];
    }

   
    // Wait until the entire tile has been loaded
  

    __syncthreads();

   
    // Compute the stencil
    

    if (row > 0 &&
        row < width - 1 &&
        col > 0 &&
        col < width - 1)
    {
        output[row * width + col] =
            (
                input_s[ty][tx] +
                input_s[ty][tx - 1] +
                input_s[ty][tx + 1] +
                input_s[ty - 1][tx] +
                input_s[ty + 1][tx]
            ) / 5.0f;
    }
}
