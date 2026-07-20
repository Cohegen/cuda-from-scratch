/*
Naive 1D stencil.
*/

#include <cuda_runtime.h>

#define STENCIL_1D_RADIUS 3
#define STENCIL_1D_WIDTH (2 * STENCIL_1D_RADIUS + 1)

__global__ void stencil1DNaive(
    const float *input,
    const float *coefficients,
    float *output,
    int size
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size)
    {
        return;
    }

    float value = 0.0f;

    for (int offset = -STENCIL_1D_RADIUS; offset <= STENCIL_1D_RADIUS; ++offset)
    {
        int inputIndex = index + offset;

        if (inputIndex >= 0 && inputIndex < size)
        {
            value += input[inputIndex] *
                coefficients[offset + STENCIL_1D_RADIUS];
        }
    }

    output[index] = value;
}
