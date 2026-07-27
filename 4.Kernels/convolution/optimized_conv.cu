/*
2D convolution using constant-memory filter coefficients and shared-memory
input tiles.
*/

#include <cuda_runtime.h>

#define OPT_TILE_WIDTH 16
#define OPT_MASK_WIDTH 3
#define OPT_MASK_RADIUS (OPT_MASK_WIDTH / 2)
#define OPT_SHARED_WIDTH (OPT_TILE_WIDTH + OPT_MASK_WIDTH - 1)

__constant__ float optimizedFilter[OPT_MASK_WIDTH * OPT_MASK_WIDTH];

void copyOptimizedFilterToConstantMemory(const float *hostFilter)
{
    cudaMemcpyToSymbol(
        optimizedFilter,
        hostFilter,
        OPT_MASK_WIDTH * OPT_MASK_WIDTH * sizeof(float)
    );
}

__global__ void optimizedConvolution2D(
    const float *input,
    float *output,
    int width,
    int height
)
{
    __shared__ float inputTile[OPT_SHARED_WIDTH][OPT_SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outputCol = blockIdx.x * OPT_TILE_WIDTH + tx;
    int outputRow = blockIdx.y * OPT_TILE_WIDTH + ty;

    for (int row = ty; row < OPT_SHARED_WIDTH; row += blockDim.y)
    {
        for (int col = tx; col < OPT_SHARED_WIDTH; col += blockDim.x)
        {
            int inputRow = blockIdx.y * OPT_TILE_WIDTH + row - OPT_MASK_RADIUS;
            int inputCol = blockIdx.x * OPT_TILE_WIDTH + col - OPT_MASK_RADIUS;

            inputTile[row][col] =
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

    for (int filterRow = 0; filterRow < OPT_MASK_WIDTH; ++filterRow)
    {
        for (int filterCol = 0; filterCol < OPT_MASK_WIDTH; ++filterCol)
        {
            value += inputTile[ty + filterRow][tx + filterCol] *
                optimizedFilter[filterRow * OPT_MASK_WIDTH + filterCol];
        }
    }

    output[outputRow * width + outputCol] = value;
}
