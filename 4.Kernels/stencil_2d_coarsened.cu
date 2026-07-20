/*
2D stencil with thread coarsening along the x dimension.
*/

#include <cuda_runtime.h>

#define STENCIL_2D_COARSE_RADIUS 1
#define STENCIL_2D_COARSE_WIDTH (2 * STENCIL_2D_COARSE_RADIUS + 1)
#define STENCIL_2D_COARSE_FACTOR 4

__global__ void stencil2DCoarsened(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int startCol =
        (blockIdx.x * blockDim.x + threadIdx.x) * STENCIL_2D_COARSE_FACTOR;

    if (row >= height)
    {
        return;
    }

    for (int item = 0; item < STENCIL_2D_COARSE_FACTOR; ++item)
    {
        int col = startCol + item;

        if (col >= width)
        {
            return;
        }

        float value = 0.0f;

        for (int filterRow = -STENCIL_2D_COARSE_RADIUS; filterRow <= STENCIL_2D_COARSE_RADIUS; ++filterRow)
        {
            for (int filterCol = -STENCIL_2D_COARSE_RADIUS; filterCol <= STENCIL_2D_COARSE_RADIUS; ++filterCol)
            {
                int inputRow = row + filterRow;
                int inputCol = col + filterCol;

                if (inputRow >= 0 && inputRow < height &&
                    inputCol >= 0 && inputCol < width)
                {
                    int filterIndex =
                        (filterRow + STENCIL_2D_COARSE_RADIUS) * STENCIL_2D_COARSE_WIDTH +
                        (filterCol + STENCIL_2D_COARSE_RADIUS);

                    value += input[inputRow * width + inputCol] *
                        coefficients[filterIndex];
                }
            }
        }

        output[row * width + col] = value;
    }
}
