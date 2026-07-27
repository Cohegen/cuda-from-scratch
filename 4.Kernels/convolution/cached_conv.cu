/*
2D convolution with the input tile cached in shared memory.
*/

#include <cuda_runtime.h>

#define CACHED_TILE_WIDTH 16
#define CACHED_MASK_WIDTH 3
#define CACHED_MASK_RADIUS (CACHED_MASK_WIDTH / 2)
#define CACHED_SHARED_WIDTH (CACHED_TILE_WIDTH + CACHED_MASK_WIDTH - 1)

__global__ void cachedConvolution2D(
    const float *input,
    const float *filter,
    float *output,
    int width,
    int height
)
{
    __shared__ float inputTile[CACHED_SHARED_WIDTH][CACHED_SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outputCol = blockIdx.x * CACHED_TILE_WIDTH + tx;
    int outputRow = blockIdx.y * CACHED_TILE_WIDTH + ty;
    int tileCol = outputCol - CACHED_MASK_RADIUS;
    int tileRow = outputRow - CACHED_MASK_RADIUS;

    for (int row = ty; row < CACHED_SHARED_WIDTH; row += blockDim.y)
    {
        for (int col = tx; col < CACHED_SHARED_WIDTH; col += blockDim.x)
        {
            int inputRow = blockIdx.y * CACHED_TILE_WIDTH + row - CACHED_MASK_RADIUS;
            int inputCol = blockIdx.x * CACHED_TILE_WIDTH + col - CACHED_MASK_RADIUS;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width)
            {
                inputTile[row][col] = input[inputRow * width + inputCol];
            }
            else
            {
                inputTile[row][col] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (outputRow >= height || outputCol >= width)
    {
        return;
    }

    float value = 0.0f;

    for (int filterRow = 0; filterRow < CACHED_MASK_WIDTH; ++filterRow)
    {
        for (int filterCol = 0; filterCol < CACHED_MASK_WIDTH; ++filterCol)
        {
            value += inputTile[ty + filterRow][tx + filterCol] *
                filter[filterRow * CACHED_MASK_WIDTH + filterCol];
        }
    }

    output[outputRow * width + outputCol] = value;
}
