/*
This is a 2D implementation of a  stencil but this 
is a naive version where we use global memory access
*/

__global__ void stencil_2d(float*input,float*output,int width)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int height = width; // Assuming a square tensor for simplicity as in the python file

    if(row < height && col < width)
    {
        int idx = row*width+col;

        float center = input[idx];
        float left = (col > 0) ? input[idx - 1] : center;
        float right = (col < width - 1) ? input[idx + 1] : center;
        float top = (row > 0) ? input[idx - width] : center;
        float bottom = (row < height - 1) ? input[idx + width] : center;

        output[idx] = (
            center + left + right + top + bottom
        )/5.0f;
    }
}