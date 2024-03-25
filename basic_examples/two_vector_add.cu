// this is a C++ program that uses CUDA to run on the GPU
// nvcc two_vector_add.cu -o add_two_vectors
// produces a.out
// run with ./add_two_vectors
// cuda does this faster than CPU
// takes less 1/3 the time of CPU

#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000
#define ops_per_thread 10

// this time, make each thread compute several elements of the array
__global__ void add(int *a, int *b, int *c)
{
    // index takes into account number of blocks and threads
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // our array is arranged in a linear fashion of
    // size N <= gridDim.x * blockDim.x
    // iterate N by index
    // we skip ever blockdim * griddim elemts
    // basically says, we add every thread i of
    //      each grid[m], block[n],
    for (int i = index; i < N; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *h_a, *h_b, *h_c;

    // perks using C++, no need to write to and from device
    // cudaMallocManaged uses Unifed Memory
    cudaMallocManaged(&h_a, N * sizeof(int));
    cudaMallocManaged(&h_b, N * sizeof(int));
    cudaMallocManaged(&h_c, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    clock_t d_begin = clock();

    int num_blocks, size_of_block;
    size_of_block = 64;
    num_blocks = (N + size_of_block - 1) / size_of_block;

    add<<<num_blocks, size_of_block>>>(h_a, h_b, h_c);
    cudaDeviceSynchronize();

    clock_t d_end = clock();

    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    // }

    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    clock_t h_begin = clock();
    for (int i = 0; i < N; i++)
    {
        h_c[i] = h_a[i] + h_b[i];
    }
    clock_t h_end = clock();

    printf("Time taken by GPU: %f\n", (double)(d_end - d_begin) / CLOCKS_PER_SEC);
    printf("Time taken by CPU: %f\n", (double)(h_end - h_begin) / CLOCKS_PER_SEC);

    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    return 0;
}
