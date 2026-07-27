/*
The naive approach of 2D convolution
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void convolution2D(
    const float* input,
    const float* filter,
    float* output,
    int width,
    int height,
    int maskWidth
)
{
    // Calculate global pixel coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread corresponds to a valid output pixel
    if (col < width && row < height)
    {
        float pixVal = 0.0f;

        // Radius of the filter
        int radius = maskWidth / 2;

        // Iterate over filter elements
        for (int mRow = 0; mRow < maskWidth; ++mRow)
        {
            for (int mCol = 0; mCol < maskWidth; ++mCol)
            {
                // Corresponding input coordinates
                int inRow = row - radius + mRow;
                int inCol = col - radius + mCol;

                // Zero-padding boundary check
                if (inRow >= 0 &&
                    inRow < height &&
                    inCol >= 0 &&
                    inCol < width)
                {
                    float pixel =
                        input[inRow * width + inCol];

                    float filterVal =
                        filter[mRow * maskWidth + mCol];

                    pixVal += pixel * filterVal;
                }
            }
        }

        // Write result to output image
        output[row * width + col] = pixVal;
    }
}