/*
Tiled 2D stencil using shared memory.
*/

#include <cuda_runtime.h>

#define STENCIL_2D_TILE_WIDTH 16
#define STENCIL_2D_TILE_RADIUS 1
#define STENCIL_2D_SHARED_WIDTH (STENCIL_2D_TILE_WIDTH + 2 * STENCIL_2D_TILE_RADIUS)
#define STENCIL_2D_FILTER_WIDTH (2 * STENCIL_2D_TILE_RADIUS + 1)

__global__ void stencil2DTiled(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height
)
{
    __shared__ float tile[STENCIL_2D_SHARED_WIDTH][STENCIL_2D_SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outputCol = blockIdx.x * STENCIL_2D_TILE_WIDTH + tx;
    int outputRow = blockIdx.y * STENCIL_2D_TILE_WIDTH + ty;

    for (int tileRow = ty; tileRow < STENCIL_2D_SHARED_WIDTH; tileRow += blockDim.y)
    {
        for (int tileCol = tx; tileCol < STENCIL_2D_SHARED_WIDTH; tileCol += blockDim.x)
        {
            int inputRow = blockIdx.y * STENCIL_2D_TILE_WIDTH + tileRow - STENCIL_2D_TILE_RADIUS;
            int inputCol = blockIdx.x * STENCIL_2D_TILE_WIDTH + tileCol - STENCIL_2D_TILE_RADIUS;

            tile[tileRow][tileCol] =
                (inputRow >= 0 && inputRow < height &&
                 inputCol >= 0 && inputCol < width)
                    ? input[inputRow * width + inputCol]
                    : 0.0f;
        }
    }

    __syncthreads();

    if (outputRow >= height || outputCol >= width)
    {
        return;
    }

    float value = 0.0f;

    for (int filterRow = 0; filterRow < STENCIL_2D_FILTER_WIDTH; ++filterRow)
    {
        for (int filterCol = 0; filterCol < STENCIL_2D_FILTER_WIDTH; ++filterCol)
        {
            value += tile[ty + filterRow][tx + filterCol] *
                coefficients[filterRow * STENCIL_2D_FILTER_WIDTH + filterCol];
        }
    }

    output[outputRow * width + outputCol] = value;
}
