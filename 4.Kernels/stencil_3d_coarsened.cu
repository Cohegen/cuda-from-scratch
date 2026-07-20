/*
3D stencil with thread coarsening along the x dimension.
*/

#include <cuda_runtime.h>

#define STENCIL_3D_COARSE_RADIUS 1
#define STENCIL_3D_COARSE_WIDTH (2 * STENCIL_3D_COARSE_RADIUS + 1)
#define STENCIL_3D_COARSE_FACTOR 2

__global__ void stencil3DCoarsened(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height,
    int depth
)
{
    int startX =
        (blockIdx.x * blockDim.x + threadIdx.x) * STENCIL_3D_COARSE_FACTOR;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (y >= height || z >= depth)
    {
        return;
    }

    for (int item = 0; item < STENCIL_3D_COARSE_FACTOR; ++item)
    {
        int x = startX + item;

        if (x >= width)
        {
            return;
        }

        float value = 0.0f;

        for (int dz = -STENCIL_3D_COARSE_RADIUS; dz <= STENCIL_3D_COARSE_RADIUS; ++dz)
        {
            for (int dy = -STENCIL_3D_COARSE_RADIUS; dy <= STENCIL_3D_COARSE_RADIUS; ++dy)
            {
                for (int dx = -STENCIL_3D_COARSE_RADIUS; dx <= STENCIL_3D_COARSE_RADIUS; ++dx)
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
                            ((dz + STENCIL_3D_COARSE_RADIUS) * STENCIL_3D_COARSE_WIDTH +
                             (dy + STENCIL_3D_COARSE_RADIUS)) * STENCIL_3D_COARSE_WIDTH +
                            (dx + STENCIL_3D_COARSE_RADIUS);

                        value += input[inputIndex] * coefficients[filterIndex];
                    }
                }
            }
        }

        output[(z * height + y) * width + x] = value;
    }
}
