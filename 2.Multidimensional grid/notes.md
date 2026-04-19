# Introduction to the module
- In the last module we looked at how we can lauch a one-dimensional grid of threads by calling a kernel function to operate on elements of a one-dimensional arrays.
- However, what is we're dealing with multidimensional data would we still launch a  one-dimensional grid of threads?
- No that wouldn't even be possible.
- In this module you'll learn how threads are organized and learn how threads and blocks can be used to process multidimensional arrays.

## 1.Multidimensional grid organization
- In CUDA, all threads in a grid execute the same kernel function, and they rely on coordinates i.e thread indices, to distinguish themselves from each other to identify the appropriate portion of the data to process.
- From our previous module we learned that a grid constists of one or more blocks, and each block consists of one or more threads.
- All threads in a block share the same block index, which can be accessed via **blockIdx** variable.
- Each threads within the block has its unique index which can be accessed via the **threadIdx** variable.
- When a thread executes a kernel function, references to the **blockIdx** and **threadIdx** variables return the coordinates of the thread.
- A grid is a 3D array of block, and each blocks is 3D array of threads.
- Now when calling the kernel we need to specifiy the size of the gird and blocks in each dimension.
- Let's have a recap on what we learned in module 1.
- We know that in order for us to specify things like size of the grid and block dimensions, we specify them using the execution configuration parameters.
- The first execution configuration parameter specifies the dimensions of the grid in  number of blocks.
- The second specifies the dimension of each block in the number of threads.
- Each parameter has the type **dim3** which is an integer vector type of three elemetns **x**, **y** and **z**.
- These three elements specify the sizes of the three dimensions.
- In this example, the following code can be used to call the **vecAddkernel()** kernel function and generate a 1D grid that consists of 32 blocks, each of which consists of 128 threads.
- Then this will make the total number of threads in the grid to be 128*32 = 4096.
```
dim3 dimGrid(32,1,1)
dim3 dimBlock(128,1,1)
vecAddkernel <<<dimGrid, dimBlock>>>(...);
```
- The grid and block dimensions can also be calculated from other variables.
```
dim3 dimGrid(ceil(n/256.0),1,1);
dim3 dimBlock(256,1,1);
vecAddkernel <<<dimGrid, dimBlock>>> (...);

```
- This allows the number of blocks to vary with the size of the vectors so that the grid will have enough threads to cover all vector elements.
- In the code above we can see that we have to fix the block size at 256.
- The value of the variable n at the kernel call tme will determine the dimension of the grid i.e the number of blocks within the grid.
- If say n is equal to 1000, the grid will consist of four blocks.
- This would enable the threads to cover a vector of maximum size of 1024 elements.

### 1.1 The **1D shortcut** for kernel launch
- Previously when describing the dimensionality of our grid and threadblocks
we did that as follows:
```
dim3 dimGrid(n,1,1);
dim3 dimBlock(256,1,1);
vecAddkernel<<<dimGrid,dimBlock>>>(...);
```
- where **n** is the number of blocks we wish out grid to have.
- However, CUDA allows a shortcut for 1D cases:
```
kernel <<<dimGrid,dimBlock>>>(..);
```
- Say we write something like this:
```
kernel <<< 16,256>>>();
```
- What CUDA does internally is that it interprets it as:
```
dim3 dimGrid(16,1,1);
dim3 dimBlock(256,1,1);
```
- Within the kernel function, the **x** field of the variables **gridDim** and **blockDim** are preinitalized according the values of the execution configuration parameters.
- If n is equal to 4000, refernces to **gridDim.x** and **dimBlock.x** in the **vecAddkernel** kernel will result in 16 and 256 respectively.

### 1.2 Why does the **1D** shortcut work?
- If you have some basic knowledge of C++ you maybe familiar with structs and constructors.
- CUDA sues a **dim3** struct type as follows:
```
struct dim
{
    unsigned int x,y,z;
}
```
- And it also has a constructor like:
```
dim3(unsigned int x =1, unsigned int y = 1, unsigned int z = 1);
```
- So when intialize **dim3** as follows:
      - **dim3(16)** it becomes **(16,1,1)**.
      - **dim3(8,4)** it becomes **(8,4,1)**.

### 1.3 Builti-in variables inside kernels
- CUDA gives us automatic variables.
      -1. **Grid Size**: **gridDim.x ,gridDim.y, gridDim.z**
      - 2 **Block size** : **blockDim.x, blockDim.y, blockDim.z**
      - 3 **Thread position** : **threadIdx.x,threadIdx.y, threadIdx.z**
      - 4 **Block position** : **blockIdx.x, blockIdx.y, blockIdx.z**

- In CUDA C the allowed values of **gridDim.x** range from 1 to (2^31) -1, and those of **gridDim.y** and **grdiDim.z** range from 1 to (2^16) -1 (65,535).
- All threads in a block share the same **blockIdx.x , blockIdx.y** and **blockIdx.z** values.
- Among blocks, the **blockIdx.x** values ranges from 0 to **gridDim.x-1**, the **blockIdx.y** value ranges from 0 to **gridDim.y-1** and the **blockIdx.z** value ranges from 0 to **gridDim.z-1**.

### 1.4  Block Configuration
- Each block is organized into a 3D array of treads.
- Two-dimensional blocks can be created by setting **blockDim.z** to 1 , as in the **vecAddkernel** example.
- All blocks in a grid have the same dimensions and sizes.
- The number of threads in each dimension of a block is specified by the second execution configuration parameter in the kernel call.
- Within the kernel call this configuration parameter can be accessed as the **x,y** and **z** field of **blockDim**.
- The total size of a block in the current CUDA systems is limited to 1024 threads.
- These threads can be distributed across three dimensions in any way as long as the total number of threads does not exceed 1024.

**ADD Grid DIAGRAM HERE!!**

- A grid and its blocks do not need the same dimensionality.
- A grid can have higher dimensionality that its blcoks and vice versa.
- For example in the above diagram, we see a grid with **gridDim** of (2,2,1) and a **blockDim** of (4,2,2).
- Such a grid would be created as the following host code:
```
dim3 dimGrid(2,2,1);
dim3 dimBlock(4,2,2);
KernelFunction <<<dimGrid, dimBlock>>>(...);
```
- The each  grid consists of four of blcks organized into a 2x2 array.
- Each block is labeled with **(blockIdx.y,blockIdx.x)**.
- For example the block (1,0) has **blockIdx.y** = 1 and **blockIdx.x** = 0.
- Each **threadIdx** also constists of three field: x coordinate **threadIdx.x**, y coordinate **threadIdx.y** and the z coordinate **threadIdx.z**.
- So if we have a grid of shape (2,2,1), it means that there's:
      - 2 blocks in the x-direction
      - 2 blocks in y-direction
      - 1 in z-direction
- So total blocks  = 2 * 2 * 1= 4 blocks
- Also if the had block of dimension (4,2,2),it means that there's:
      - 4 threads in the x-direction
      - 2 in y-direction
      - 2 in z-direction

- The total threads per block is:
```
4 x 2 x 2 = 16 threads
````

## 2 Mapping threads to multidimensional data
- Say we have picture **P** which havs 62 pixels in the vertical or y direction and 76 pixels the horizontal or x axis.
- Assume that we decided to use a 16 x 16 block, with 16 threads in the x direction and 16 threads in the y direction.
- We will need four blocks in the y direction and five blocks in the x direction, which resulrt to 4 x 5 = 20 blocks.
- One Grid  looks like this:
```
            X direction →
        0      1      2      3      4
      +------+------+------+------+------+
Y  0  | B00  | B01  | B02  | B03  | B04  |
   1  | B10  | B11  | B12  | B13  | B14  |
   2  | B20  | B21  | B22  | B23  | B24  |
   3  | B30  | B31  | B32  | B33  | B34  |
      +------+------+------+------+------+
```

- Each block looks like this:
```
        threadIdx.x →
        0 1 2 3 ... 15
      +------------------+
   0  | o o o o ... o    |
   1  | o o o o ... o    |
   2  | o o o o ... o    |
   .  | o o o o ... o    |
  15  | o o o o ... o    |
      +------------------+
threadIdx.y ↓
```

- Each thread is assigned to process a pixel whose y and x coordinates are derived from its **blockIdx, blockDim** and **threadIdx** varaible values:
```
vertical(row) row coordinate = blockIdx.y*blockDim.y + threadIdx.y
horizontal(columns) coordinate = blockIdx.x*blockDim.x + threadIdx.x
```
- For example $P_in$ elememet to processed by thread (0,0) of block(1,0) can be identified as follows:
```
Pin(blockIdx.y*blockDim.y + threadIdx.y,blockIdx.x * blockDim.x+threadIdx.x) 
= Pin(1* 16+0, 0* 16+0) = Pin(16,0)

```
- Recall from module 1 that, an if-statement is needed to prevent any extra thread from taking effect.
- Here in the case of picture processing we can expect that the kernel function will have if-statements to test whether the threads's vertical and horizontal indices fall within the valid range of pixels.
- Here we assume that the host code uses an integer variable **n** to track the number of pixels in the **y** direction and another integer variable **m** to track the number of pixels in the **x** direction.
- We assume that the input picture has been copied into the device global memory and can be accessed through a pointer variable **Pin_d**.
- The output picture has allocated in the device memory and can be accessed through a pointer variable **Pout_d**.
- The host code below can be use to call a 2D kernel **colorToGrayscaleConversion** to process the picture as follows:
```
dim3 dimGrid(ceil(m/16.0),ceil(n/16.0),1);
dim3 dimBlock(16,16,1);
colorToGrayscaleConversion <<<dimGrid,dimBlock>>>(Pin_d,Pout_d,m,n);

```
- In this example, we assume that the dimensions of the blocks are fixed at 16 x 16.
- The dimensions of the grid, on the other hand, depend on the dimensions of the picture.
- Say we want to process a 1500x 2000 (3 million pixel) picture, we would generate 11,750 blocks: 94 in the y direction and 125 in the x direction.
- Within the kernel function, references to **gridDim.x,gridDim.y,blockDim.x** and **blockDim.y** will result in 125,94,16 and 16 respectively.

### 2.1 How C statements acess elements of dynamically allocated multidimensional arrays.
- We need a way of accessing **Pin_d** as a 2D array whereby in which an element at row j and column i can be accessed as **Pin_d[j][i]**.
- The ANSI C standard on the basis of which CUDA C was developed requires the number of columns in **Pin** to be known at compile time for **Pin** to be accessed as a 2D array.
- However, this information is not known at compile time for dynamically allocated arrays.
- The reason why at times dynamically allocated arrays as used is to allow the sizes and dimensions of these to varay according to the data size at runtime.
- So information on the number of columns in a dynamically allocated array is unknown at compile time dy design.
- Due to this , we as programmers need to linearize or **flatten** a dynamically allocated 2D array into its equivalent 1D array.
- All multidimensional arrays in C are linearized.
- This is due to the use of a **flat** memory space in modern computers.
- Under the hood in the computer, the compiler linearizes the multi-dimensional arrays into their equivalent  1D array and translates the mulidimensional index syntax into a 1D offset.

#### 2.1.1 Ways of Linearizing 2D arrays
- There are two ways which a 2D array can be linearized.
- One is to place all elements of the same row into consecutive locations.The rows are then placed one after anther into the memory space. This arrangement is called **row major layout**.
- The example of this is shown below:
```
   col →
     0   1   2   3
row
 0  [a   b   c   d]
 1  [e   f   g   h]
 2  [i   j   k   l]
```
- Above the have a 2D Matrix, but we wish to flatten it.
- To do this we perform **row-major**:
```
[a  b  c  d  e  f  g  h  i  j  k  l]
 0  1  2  3  4  5  6  7  8  9  10 11   ← memory index
```
- To improve readabiliy, we use $M_{j,i}$ to denote an element **M** of the **jth** row and the **ith** column.
- Therefore the 1D equivalent for an element of  **M** at row j and column i is j*4 + i.
- The j*4 terms skips over all elements of the rows before row j.
- The i term the selects the right element  within the section for row j.
- For example  the 1D  index for $M_{2,1}$ is 2*4 + 1 = 9.
- So the general formula for the index of an element at position (row,col) :
   - index = row * width + col


- The second way to linearize a 2D array is to place all elements of the same column in consecutive locations.
- The columns are then placed one after another into the memory space.
- This arrangement is called **column-major**.

```
//The input image is encoded as unsigned chars  [0,255]
//Each pixel is 3 consecutive chars for the 3 channels (RGB)

__global__
void colorToGrayscaleConversion(unsigned char *Pout, unsigned char *Pin,int width,int height)
{
      int col = blockIdx.x*blockDim.x + threadIdx.x;
      int row = blockIdx.y*blockDim.y + threadIdx.y;
      if(col <width && row < height)
      {
            //Getting 1D offset for the grayscale image
            int grayOffset = row*width + col;

             //RGB Can be thought of having many channels
             int rgbOffset  = grayOffset*CHANNELS;
             unsigned char r = Pin[rgbOffset ]; //Red value
             unsigned char g = Pin[rgbOffset + 1]; // green value
             unsigned char b = Pin[rgbOffset + 2]; //blue value

             //Performing the rescaling and storing it
             //We multiply by floating point constants
             Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
      }
}
```

- The kernel code above uses the following equation to convert each pixel to it grayscale counterpart:
```
 L - 0.21*r + 0.72*g + 0.07*b
```

- There are a total of blockDim.x*gridDim.x threads in the horizontal direction. 
- The expression below generates every integer value from 0 to blockDim.x*blockDim.x-1.

```
col = blockIdx.x*blockDim.x + threadIdx.x
```
- We know that gridDim.x * blockDim.x is greater than or equal to width (m value passed in from the host code).
- We have at least as many threads as the number of pixels in the horizontal direction.
- We also know that there are at least as many threads as the number of pixels  in the vertical direction.
- So we need a way to test and make sure that onlt the threads with both row and column values are within range, i.e (col < width) &&  ( row < height).
- Since there are **width** pixels in each row, we can generate the 1D index for the pixel at row **row** and column **col** as row*width + col.
- This 1D index **grayOffset** is the pixel index for **Pout** since each pixel in the output grayscale image is 1 byte (unsigned char).
- Using a good example of a 62 x 76 image example, the linearized 1D  index of **Pout** pixel is calculated by thread(0,0) of block (1,0) with the following formula:
```
Pout(blockIdx.y*blockDim.y+threadIdx.y,blockIdx.x*blockDim.x+threadIdx.x)
 =Pout(1*16+0,0*16+0) = Pout(16,0) =
 Pout[16*76 + 0] = Pout[1216]


```

- For **Pin** we need multiply the gray pixel index vy 32F32F sincie each colored ixel is stores as three elemets (r,g,b) each of which is 1 byte.
- The resulting **rgbOffset** gives the staring location of color pixel in the **Pin** array.
- We read the r,g and b value from the three consecutive byte locations of the **Pin** array, perform the calculations of the grayscale pixel value, and write the value into **Pout** array using **grayOffset**.
- In out 62x76  image examplethe linearized 1D index of the first componet of the **Pin** pexel that is processed by thread (0,0) of block (1,0) can be calculated with the following formula :
```
Pin(blockIdx.y*blcokDim.y+threadIdx.y, blockIdX.x*blockDim.x+threadIdx.x) =
Pin(1*16+0,0*16+0) = Pin(16,0) =Pin[16*76*3+0] = Pin[3648]
```

## 3 Image Blur ( a more complex kernel)
- In reality CUDA C programs, threads often perform complex operations on their data and need to cooperate with each other.
- Image blurring smoothes out abrupt variation of pixel values while preserving the edges that are essential for recognizing the key features of the image.
- To the human eyes, a blurred image tends to obscure the fine details and present the **big picture** impression,of the major thematic objects in the picture.
- In computer image-processing algorithms a common use case of image blurring is to reduce the impact of noise and granular rendering effects in an image by correcting problematic pixel values within the clean surrounding pixel values.
- In computer vision, bluring can be used to allow edge detection and object recognition algorithms to focus on thematic objects rather than being bogged down by a massive quantity of  fine-grained objects.
- Mathematically, an image blurring function calculates the value of an output image pixel as a weighted sum of a patch of pixels encompassing the pixel in the input image.

### 3.1 Image blur example
- In the example we use 3x3 patch.
- When calculating an output pixel value at (row,col) position, we see that the patch is centered at the input pixel located at the (row,col) position.
- The 3x3 patch spans three rows (row-1,row,row+1) and three columns (col-1,col,col+1).

```
            Columns →
           col-1   col   col+1
         +-------+------+-------+
row-1    |(r-1,c-1)|(r-1,c)|(r-1,c+1)|
         +-------+------+-------+
row      |(r,c-1)  |(r,c) |(r,c+1) |
         +-------+------+-------+
row+1    |(r+1,c-1)|(r+1,c)|(r+1,c+1)|
         +-------+------+-------+
```
- For example, the coordinates of the nine pixels for calculating the output pixel at (25,50) are (24,49),(24,51),(25,49),(25,50),(25,51),(26,49),(26,50) and (26,51).
```
            Columns →
             49      50      51
         +-------+-------+-------+
24       |(24,49)|(24,50)|(24,51)|
         +-------+-------+-------+
25       |(25,49)|(25,50)|(25,51)|
         +-------+-------+-------+
26       |(26,49)|(26,50)|(26,51)|
         +-------+-------+-------+
```


```
__global__
void blurKernel(unsigned char * in, unsigned char *out, int w,int h)
{
      int col = blockIdx.y*blockDim.y + threadIdx.y;
      int row = blockIdx.x*blockDim.x + threadIdx.x;

      if (row < w && col <  h)
      {
            int pixVal = 0;
            int pixels = 0;

            //getting the average of the surrounding BLUR_SIZE X BLUR_SIZE box
            for(int blurRow=-BLUR_SIZE;blurRow<BLUR_SIZE+1;++blurRow)

            {
                  for(int blurCol=-BLUR_SIZE;blurCol<BLUR_SIZE+1;++blurCol)
                  {
                        
                  int curRow = row + blurRow;
                  int curCol = col + blurCol;

                  //verifying that we have a valid image pixel
                  if(curRow>=0 && curRow<h && curCol >=0 && curRow<w)
                  {
                        pixVal += in[curRow*w + curCol];
                        ++pixels; //keep track of number of pixels in the average
                  }
                  }
            }
            //writing our new pixel value out
            out[row*w + col] = (unsigned char)(pixVal/pixels);
      }
}
```
- In the diagram above, we see that **col** and **row** values also give the central pixel location of patch of the input pixels used for calculating the output pixel for the thread.
- The nested for-loop in the program above iterates through all the pixels in the patch.
- We assume that the program has a defined constant **BLUR_SIZE**.
- The value of **BLUR_SIZE** is set such that **BLUR_SIZE** gives the number of pixels on each side (radius) of the patch.
- For example for a 3x3 patch, **BLUR_SIZE** is set to 1, whereas for a 7x7 patch, BLUR_SIZE is set to 3.
- The outer loop iterates through the rows of the patch while the inner loop iterates through the columns of the patch.
- In our example where we use a 3x3 patch example, the **BLUR_SIZE** is 1.
- For the thread that calculates output pixel (25,50) during the first iteration of the outer loop, the **curRow** variable is **row-BLUR_SIZE** = (25-1) = 24.
- Thus during the first iteration of the outer loop, the inner loop iterates through the patch pixels in row 24.
- The inner loop iterates from column **col-BLUR_SIZE** = 50-1 =49 to **col+BLUR_SIZE** = 51 using the curCol variable.
- Therefore the pixels that are processes in the first iteration of the outer are (24,29),(24,50) and (24,51).
- We then use a linearized index of **curRow** and **curCol** i.e **pixVal+= in[curRow*w + curCol];** to access the value of the input pixel visited in the current iteration.
- It accumulates the pixel value into a running sum variable**pixVal**.
- After this the average is calculated by **out[row*w + col] = (unsinged char) (pixVal /pixels);**.


## 2. Matrix-Matrix Multiplication
- Matrix multiplication between an **I x j** (i rows by j columns) matrix **M** and a **jxk** matrix **N** produces an **Ixk** matrix **P**.
- When a matrix multiplication is performed, each element of the output matrix **P** is an inner dot producet of rows of M and columns of N.
- To implement matrix multiplication using CUDA, we can map the threads in the grid to the elements of the output matrix **P** with the same approach used for **colorToGrayScaleConversion**.
- That is, each thread is responsible for calculating one element of P.
- The row and columns indices for the P to be calculated by each thread are the same as before:
```
row = blockIdx.y*blockDim.y + threadIdx.y

col = blockIdx.x*blockDim.x + threadIdx.x
```
- The source code to perform MATMUL is shown below:
```
__global__ void matmulKernel(float *M,float *N, float *P, int width)
{
      int row = blockIdx.y*blockDim.y + threadIdx.y
      int col = blockIdx.x*blockDim.x + threadIdx.x

      if(row < width ) && (col <  width)
      {
            float Pvalue = 0
            for(int k =0;i< width; ++k)
            {
                  Pvalue += M[row*width+k] *N[k*width+col];
            }
            P[row*width+col] = Pvalue;
      }
}
```
- $P_{row,col}$ is calculated as the inner product of the rowth row of M and the colth column of N.
- In the above code snippet we use a for-loop to perform this operation.
- Before the loop,we initialize a local variable **Pvalue** to 0.
- Each iteration of the loop access an element from the rowth row of M and an element from the colth column of N, which multiplies two elements together, and accumulates the product into **Pvalue**.
- Matrix M is linearized in row major order, starting from the 0th row.
- Therefore the beginning elemet of the row 1 is **M[1*width]** because we need to account for all elements of row 0.
- In general the beginning element is the rowth row is **M[row*width]**.
- Since all elements of a row are placed in a consecutiv locations, the kth element of the rowth row is at **M[row*width+k].

- We now turn our focus to accessing the elements in matrix N.
- The beginning element of the colth element is the colth element of row 0, which is N[col].
- Accessing the next element in the colth column requires skipping over an entire row.
- This is because the next element of the same rwo is the same element in the next row.
- Therefore the kth element in the colth column is **N[k*Width+col]**.
- After execution exists the for loop, all threads have their P element is the Pvalue variables.
- Each thread then used the 1D equivalent index expression **row*width+col** to write its **P** element.

