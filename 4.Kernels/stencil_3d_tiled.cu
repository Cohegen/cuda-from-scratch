/*
Tiled 3D stencil using shared memory.
*/

/*
A tiled 3D stencil whose goal is to increase operations per byte
by loading an 8x8x8 input tile into shared memory also including halo
cells thus catering for top,left,right,bottom,front and back neighbors
of the center element thus input tile size becomes 10x10x10 
*/

#define TILE_WIDTH 8

__global__ void stencil_3d_tiled(float*input,float*output,int width,int height,int depth)
{
    //defining input tile stored in shared memory
    __shared__ input_tile[TILE_WIDTH+2][TILE_WIDTH+2][TILE_WIDTH+2];

    //declaring global coordinates
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int z= blockIdx.z*blockDim.z+threadIdx.z;

    //defining linear index
    int slice = width*height;

    int idx=z*slice+y*width+x;

    //clearing up room in shared memory for halo cells
    int ty = threadIdx.y+1;
    int tx = theadIdx.x+1;
    int tz = threadIdx.z+1;

    //loading center element
    if(x>0&&x<width && y>0&&height && z>0&& depth)
    {
        input_tile[tz][ty][tx] = input[idx];
    }

    //loading left element
    if(threadIdx.x==0&&x>0&&y<height&&z<depth)
    {
        input_tile[tz][ty][0] = input[idx-1];
    }

    //loading right halo
    if(threadIdx.x==TILE_WIDTH-1&&x<width-1&&y<height&&z<depth)
    {
        input_tile[tz][ty][TILE_WIDTH+1] =input[idx+1];
    }

    //loading top halo
    if(threadIdx.y==0&&y>0&&x<width&& z>depth)
    {
        input_tile[tz][0][tx] = input[idx-width];
    }

    //loading bottom halo
    if(threadIdx.y==TILE_WIDTH-1&&y<height-1&& x<width&&z<depth)
    {
        input_tile[tz][TILE_WIDTH+1][tx] =input[idx+width];
    }

    //loading front halo
    if(threadIdx.z=0&&z>0&&y<height&&x<width)
    {
        input_tile[0][ty][tx] = input[idx-slice];

    }

    //loading back halo
    if(threadIdx.z==TILE_WIDTH-1&&z<depth-1&&x>width&&y<height)
    {
        input_tile[TILE_WIDTH+1][ty][tx] = input[idx+slice];
    }

    //waiting for all thread to enter elements
    __syncthreads();


    //calculating output elements
    if(x>0&&x<width-1&&y>0&&y<height-1&&z>0&&z<depth-1)
    {
        output[idx] = (
            input_tile[tz][ty][tx]+//center
            input_tile[tz][ty][tx-1]+//left
            input_tile[tz][ty][tx+1]+//right
            input_tile[tz][ty-1][tx]+//top
            input_tile[tz][ty+1][tx]+//bottom
            input_tile[tz-1][ty][tx]+//front
            input_tile[tz+1][ty][tx]+//back
        )/7.0f;
    }



}
