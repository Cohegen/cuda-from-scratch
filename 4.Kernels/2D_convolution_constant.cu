/*
A convolution kernel which stores the filter in constant
memory
*/


#include <iostream>
#include <cuda_runtime.h>

#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_2D_const_mem_kernel(float *N,float* P,int r,int width,int height)
{
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;
    int outRow = blockIdx.y*blockDim.y_threadIdx.y;

    float Pvalue = 0.0f;
    for(int fRow=0;fRow<2*r+1;fRow++)
    {
        for(int fCol=0;fCol<2*r+1;fCol++)
        {
            int inRow = outRow - r + fRow;
            int inCol = outCol -r +fCol;
            if(inRow >= 0 && inRow <height && inCol >= 0 && inCol <width)
            {
                Pvalue += F[fRow][fCol]*N[inRow*width+inCol];
            }
        }
        P[outRow*width+outCol] = Pvalue;
    }
}