#include <cuda_runtime.h>

#define TILE_WIDTH     16
#define COARSE_FACTOR   2

__global__
void stencil_tiled_coarsened(
    const float* input,
    float* output,
    int width)
{
    __shared__
    float tile[TILE_WIDTH + 2]
              [TILE_WIDTH * COARSE_FACTOR + 2];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Beginning of this thread's work
    int row = blockIdx.y * TILE_WIDTH + ty;
    int baseCol =
        blockIdx.x * (TILE_WIDTH * COARSE_FACTOR)
        + tx;

    //----------------------------------------------------
    // Load center values
    //----------------------------------------------------

    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int col = baseCol + c * TILE_WIDTH;

        if (row < width && col < width)
        {
            tile[ty + 1][tx + c * TILE_WIDTH + 1] =
                input[row * width + col];
        }
    }

    //----------------------------------------------------
    // Left halo
    //----------------------------------------------------

    if (tx == 0)
    {
        int col = baseCol;

        if (row < width && col > 0)
        {
            tile[ty + 1][0] =
                input[row * width + col - 1];
        }
    }

    //----------------------------------------------------
    // Right halo
    //----------------------------------------------------

    if (tx == TILE_WIDTH - 1)
    {
        int col =
            baseCol +
            (COARSE_FACTOR - 1) * TILE_WIDTH;

        if (row < width &&
            col < width - 1)
        {
            tile[ty + 1]
                [TILE_WIDTH * COARSE_FACTOR + 1] =
                input[row * width + col + 1];
        }
    }

    //----------------------------------------------------
    // Top and bottom halos
    //----------------------------------------------------

    if (ty == 0)
    {
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++)
        {
            int col = baseCol + c * TILE_WIDTH;

            if (row > 0 &&
                col < width)
            {
                tile[0][tx + c * TILE_WIDTH + 1] =
                    input[(row - 1) * width + col];
            }
        }
    }

    if (ty == TILE_WIDTH - 1)
    {
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++)
        {
            int col = baseCol + c * TILE_WIDTH;

            if (row < width - 1 &&
                col < width)
            {
                tile[TILE_WIDTH + 1]
                    [tx + c * TILE_WIDTH + 1] =
                    input[(row + 1) * width + col];
            }
        }
    }

    //----------------------------------------------------
    // Four corners
    //----------------------------------------------------

    if (tx == 0 && ty == 0)
    {
        if (row > 0 && baseCol > 0)
            tile[0][0] =
                input[(row - 1) * width + baseCol - 1];
    }

    if (tx == TILE_WIDTH - 1 && ty == 0)
    {
        int col =
            baseCol +
            (COARSE_FACTOR - 1) * TILE_WIDTH;

        if (row > 0 &&
            col < width - 1)
        {
            tile[0][TILE_WIDTH * COARSE_FACTOR + 1] =
                input[(row - 1) * width + col + 1];
        }
    }

    if (tx == 0 && ty == TILE_WIDTH - 1)
    {
        if (row < width - 1 &&
            baseCol > 0)
        {
            tile[TILE_WIDTH + 1][0] =
                input[(row + 1) * width + baseCol - 1];
        }
    }

    if (tx == TILE_WIDTH - 1 &&
        ty == TILE_WIDTH - 1)
    {
        int col =
            baseCol +
            (COARSE_FACTOR - 1) * TILE_WIDTH;

        if (row < width - 1 &&
            col < width - 1)
        {
            tile[TILE_WIDTH + 1]
                [TILE_WIDTH * COARSE_FACTOR + 1] =
                input[(row + 1) * width + col + 1];
        }
    }

    __syncthreads();

    //----------------------------------------------------
    // Compute two stencil outputs
    //----------------------------------------------------

    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int col = baseCol + c * TILE_WIDTH;
        int lx = tx + c * TILE_WIDTH + 1;

        if (row > 0 &&
            row < width - 1 &&
            col > 0 &&
            col < width - 1)
        {
            output[row * width + col] =
            (
                tile[ty + 1][lx] +
                tile[ty + 1][lx - 1] +
                tile[ty + 1][lx + 1] +
                tile[ty][lx] +
                tile[ty + 2][lx]
            ) * 0.2f;
        }
    }
}
