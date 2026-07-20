/*
A 2D stencil kernel where instead of an individual thread is to calculate one output element
it instead calculates two or more output elements

*/

#define TILE_WIDTH 16
#define COARSE_FACTOR 2


__global__
void stencil_2d_coarsened(const float* input,
                          float* output,
                          int width,
                          int height)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;

    int rowOffset = row * width;

    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int currentCol = col + c * TILE_WIDTH;

        if (row > 0 &&
            row < height - 1 &&
            currentCol > 0 &&
            currentCol < width - 1)
        {
            int idx = rowOffset + currentCol;

            output[idx] =
            (
                input[idx] +
                input[idx - 1] +
                input[idx + 1] +
                input[idx - width] +
                input[idx + width]
            ) / 5.0f;
        }
    }
}
