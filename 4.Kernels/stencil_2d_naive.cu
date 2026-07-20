/*
This is a 2D implementation of a  stencil but this 
is a naive version where we use global memory access
*/

__global__ void stencil_2d(float*input,float*output,int width)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if((row>0 && row<width-1)&& (col>0 && col<width-1))
    {
        int idx = row*width+col;
        output[idx] = (
            input[idx]+input[idx-1]+input[idx+1]+input[idx-width]+input[idx+width]
        )/5.0f;
    }
}
