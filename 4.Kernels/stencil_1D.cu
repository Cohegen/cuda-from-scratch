/*
A 1D implementation of Stencil
This stencil averages three neighboring elements i.e on the right and one on the left
*/

__global__ void stencil_1D(float*input,float*output,int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width)
    {
        // Handle boundary conditions: use the element itself if neighbor is out of bounds
        float left = (i > 0) ? input[i - 1] : input[i];
        float center = input[i];
        float right = (i < width - 1) ? input[i + 1] : input[i];

        output[i] = (left + center + right) / 3.0f;
    }
}