/*
Naive 3D stencil using global memory.
*/

#include <cuda_runtime.h>

#define STENCIL_3D_RADIUS 1
#define STENCIL_3D_WIDTH (2 * STENCIL_3D_RADIUS + 1)

__global__ void stencil3DNaive(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height,
    int depth
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth)
    {
        return;
    }

    float value = 0.0f;

    for (int dz = -STENCIL_3D_RADIUS; dz <= STENCIL_3D_RADIUS; ++dz)
    {
        for (int dy = -STENCIL_3D_RADIUS; dy <= STENCIL_3D_RADIUS; ++dy)
        {
            for (int dx = -STENCIL_3D_RADIUS; dx <= STENCIL_3D_RADIUS; ++dx)
            {
                int inputX = x + dx;
                int inputY = y + dy;
                int inputZ = z + dz;

                if (inputX >= 0 && inputX < width &&
                    inputY >= 0 && inputY < height &&
                    inputZ >= 0 && inputZ < depth)
                {
                    int inputIndex = (inputZ * height + inputY) * width + inputX;
                    int filterIndex =
                        ((dz + STENCIL_3D_RADIUS) * STENCIL_3D_WIDTH +
                         (dy + STENCIL_3D_RADIUS)) * STENCIL_3D_WIDTH +
                        (dx + STENCIL_3D_RADIUS);

                    value += input[inputIndex] * coefficients[filterIndex];
                }
            }
        }
    }

    output[(z * height + y) * width + x] = value;
}
