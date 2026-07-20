/*
Naive 2D stencil using global memory.
*/

#include <cuda_runtime.h>

#define STENCIL_2D_RADIUS 1
#define STENCIL_2D_WIDTH (2 * STENCIL_2D_RADIUS + 1)

__global__ void stencil2DNaive(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
    {
        return;
    }

    float value = 0.0f;

    for (int filterRow = -STENCIL_2D_RADIUS; filterRow <= STENCIL_2D_RADIUS; ++filterRow)
    {
        for (int filterCol = -STENCIL_2D_RADIUS; filterCol <= STENCIL_2D_RADIUS; ++filterCol)
        {
            int inputRow = row + filterRow;
            int inputCol = col + filterCol;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width)
            {
                int filterIndex =
                    (filterRow + STENCIL_2D_RADIUS) * STENCIL_2D_WIDTH +
                    (filterCol + STENCIL_2D_RADIUS);

                value += input[inputRow * width + inputCol] *
                    coefficients[filterIndex];
            }
        }
    }

    output[row * width + col] = value;
}
