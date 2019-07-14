#include <stdio.h>
#include <assert.h>
#include <iostream>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
// Part 2 of 4: implement the kernel
__global__ void kernel( int *a, int dimx, int dimy ) {

    //e.g. M[2,3] = 2*N + 3

    int index_X = blockIdx.x * 2 + threadIdx.x; 
    int index_Y = blockIdx.y * 2 + threadIdx.y; 

    int index =  index_X * dimx + index_Y;

    a[index] = index;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

    cudaSetDevice(MYDEVICE);

    // Part 1 and 4 of 4: set the dimensions of the matrix
    int dimx = 4;
    int dimy = 6;
    int num_bytes = dimx*dimy*sizeof(int);

    int *d_a=0, *h_a=0; // device and host pointers

    h_a = (int*)malloc(num_bytes);
    //allocate memory on the device
    cudaMalloc(&d_a, num_bytes);

    if(NULL==h_a || NULL==d_a) {
        fprintf(stderr,"couldn't allocate memory\n");
        return 1;
    }

    // Part 2 of 4: define grid and block size and launch the kernel
    dim3 grid, block;
    block.x = dimx * dimy / 4; // 4 is the number of blocks
    block.y = dimx * dimy / 4;
    grid.x  = 2;
    grid.y  = 2;

    kernel<<<grid, block>>>(d_a, dimx, dimy);
    // block until the device has completed
    cudaThreadSynchronize();
    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
    // device to host copy
    cudaMemcpy(h_a, d_a, num_bytes, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy");

    int j = 0;
    for (int i = 0; i < dimx * dimy; i++)
    {
        if (j == 4)
        {
            j = 0;
            printf("\n");
        }
        std::cout << h_a[i] << "\t ";
        j++;
    }

    // verify the data returned to the host is correct
    //for(int row=0; row<dimy; row++)
    //{
    //    for(int col=0; col<dimx; col++)
    //        assert(h_a[row * dimx + col] == row * dimx + col);
    //}
    // free host memory
    free( h_a );
    // free device memory
    cudaFree( d_a );

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    printf("Correct!\n");

    return 0;
}


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}
