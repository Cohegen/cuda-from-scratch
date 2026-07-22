/*
Naive 3D stencil using global memory.
*/

/*
The 3D stencil is a 7-point stencil which uses:
      - center
      -left
      -right
      -up
      -down
      -front
      -back

It's visualization is shown below
                Front (z-1)
                   X
                   |
                   |
Up (y-1)  X ---- Center ---- X  Down (y+1)
                   |
                   |
                Back (z+1)


//--Memory Layout---
Given depth(D),height(H) and width(W), a point(z,y,x) is stored in a 1D array as:
   idx = z*(height*width)+y*width+x

So neighbors becomes:
    -left = idx-1
    -right = idx+1

    -up = idx-width
    -down = idx+width

    -front = idx -(height*width)
    -back = idx+(height*width)

    So moving :
       - one column -> +-1
       - one row -> +- width
       - one slice -> +-(height*width)
*/


#include <cuda_runtime.h>
#include <iostream>

__global__ void stencil_3d_naive(float*input,float*output,int depth,int width,int height)
{

    //golbal coordinates 
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int z = blockIdx.z*blockDim.z+threadIdx.z;

    //boundary checks
    if(x>0&&x<width-1&&y>0&&height-1&&z>0&&depth-1)
    {
        int idx = z*(height*width)+y*height+x;

        output[idx] =(
            output[idx]+//center
            output[idx-1]+//left
            output[idx+1]+//right
            output[idx-width]+//up
            output[idx+width]+//bottom
            output[idx-(height*width)]+ //front
            output[idx+(height*width)]//back
        )/7.0f;
    }
}
