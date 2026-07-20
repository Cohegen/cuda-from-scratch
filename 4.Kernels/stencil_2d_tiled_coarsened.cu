/*
Tiled 2D stencil with thread coarsening along the x dimension.
*/

#include <cuda_runtime.h>

#define STENCIL_2D_TC_TILE_WIDTH 16
#define STENCIL_2D_TC_RADIUS 1
#define STENCIL_2D_TC_FACTOR 2
#define STENCIL_2D_TC_OUTPUT_WIDTH (STENCIL_2D_TC_TILE_WIDTH * STENCIL_2D_TC_FACTOR)
#define STENCIL_2D_TC_SHARED_WIDTH (STENCIL_2D_TC_OUTPUT_WIDTH + 2 * STENCIL_2D_TC_RADIUS)
#define STENCIL_2D_TC_FILTER_WIDTH (2 * STENCIL_2D_TC_RADIUS + 1)

__global__ void stencil2DTiledCoarsened(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height
)
{
    __shared__ float tile[STENCIL_2D_TC_TILE_WIDTH + 2 * STENCIL_2D_TC_RADIUS]
                         [STENCIL_2D_TC_SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outputRow = blockIdx.y * STENCIL_2D_TC_TILE_WIDTH + ty;
    int outputColBase =
        blockIdx.x * STENCIL_2D_TC_OUTPUT_WIDTH +
        tx * STENCIL_2D_TC_FACTOR;

    for (int tileRow = ty;
         tileRow < STENCIL_2D_TC_TILE_WIDTH + 2 * STENCIL_2D_TC_RADIUS;
         tileRow += blockDim.y)
    {
        for (int tileCol = tx; tileCol < STENCIL_2D_TC_SHARED_WIDTH; tileCol += blockDim.x)
        {
            int inputRow = blockIdx.y * STENCIL_2D_TC_TILE_WIDTH +
                tileRow - STENCIL_2D_TC_RADIUS;
            int inputCol = blockIdx.x * STENCIL_2D_TC_OUTPUT_WIDTH +
                tileCol - STENCIL_2D_TC_RADIUS;

            tile[tileRow][tileCol] =
                (inputRow >= 0 && inputRow < height &&
                 inputCol >= 0 && inputCol < width)
                    ? input[inputRow * width + inputCol]
                    : 0.0f;
        }
    }

    __syncthreads();

    if (outputRow >= height)
    {
        return;
    }

    for (int item = 0; item < STENCIL_2D_TC_FACTOR; ++item)
    {
        int outputCol = outputColBase + item;

        if (outputCol >= width)
        {
            continue;
        }

        float value = 0.0f;
        int tileColBase = tx * STENCIL_2D_TC_FACTOR + item;

        for (int filterRow = 0; filterRow < STENCIL_2D_TC_FILTER_WIDTH; ++filterRow)
        {
            for (int filterCol = 0; filterCol < STENCIL_2D_TC_FILTER_WIDTH; ++filterCol)
            {
                value += tile[ty + filterRow][tileColBase + filterCol] *
                    coefficients[filterRow * STENCIL_2D_TC_FILTER_WIDTH + filterCol];
            }
        }

        output[outputRow * width + outputCol] = value;
    }
}
