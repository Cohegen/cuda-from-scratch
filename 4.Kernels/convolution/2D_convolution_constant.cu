/*
2D convolution using constant memory for the filter.
*/

#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

__constant__ float constantFilter[FILTER_WIDTH * FILTER_WIDTH];

void copyFilterToConstantMemory(const float *hostFilter)
{
    cudaMemcpyToSymbol(
        constantFilter,
        hostFilter,
        FILTER_WIDTH * FILTER_WIDTH * sizeof(float)
    );
}

__global__ void convolution2DConstant(
    const float *input,
    float *output,
    int width,
    int height
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
    {
        return;
    }

    float value = 0.0f;

    for (int filterRow = 0; filterRow < FILTER_WIDTH; ++filterRow)
    {
        for (int filterCol = 0; filterCol < FILTER_WIDTH; ++filterCol)
        {
            int inputRow = row + filterRow - FILTER_RADIUS;
            int inputCol = col + filterCol - FILTER_RADIUS;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width)
            {
                value += input[inputRow * width + inputCol] *
                    constantFilter[filterRow * FILTER_WIDTH + filterCol];
            }
        }
    }

    output[row * width + col] = value;
}
