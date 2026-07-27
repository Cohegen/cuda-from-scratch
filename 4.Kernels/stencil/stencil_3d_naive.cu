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

__global__ void stencil_3d_naive(const float* input, float* output, int width, int height, int depth)
{
    // Global coordinates 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int slice = width * height;

    // Boundary checks
    if (x > 0 && x < width - 1 &&
        y > 0 && y < height - 1 &&
        z > 0 && z < depth - 1)
    {
        int idx = z * slice + y * width + x;

        output[idx] = (
            input[idx] +                      // center
            input[idx - 1] +                  // left
            input[idx + 1] +                  // right
            input[idx - width] +              // up
            input[idx + width] +              // bottom
            input[idx - slice] +              // front
            input[idx + slice]                // back
        ) / 7.0f;
    }
    else if (x < width && y < height && z < depth)
    {
        int idx = z * slice + y * width + x;
        output[idx] = input[idx];
    }
}

