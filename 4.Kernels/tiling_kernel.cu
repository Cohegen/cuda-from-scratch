#include <iostream>
#include "cuda_runtime.h"
#include "cuda_tile.h"



//tile kernel entry point. Cannot be called directly must be launched
__tile__global__ void tileKernel(float *a,float *b,float *c)
{
    namespace ct = cuda::tiles;
    int bid_x = ct::bid().x; //block index along .x
    int bid_y = ct::bid().y; //block indexalong .y
    int num_x = ct::num_block().x; //total blocks along .x
    
}

//tile function. callable from the tile kernels and tile functions
__tile__ float helper(float x, float y)
{
    return x+y;
}

//launching the kernel
tileKernel<< dim3(num_block_x,num_blocks_y), 1 >>(a,b,c);