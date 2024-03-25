#include<iostream>
#include<cuda_runtime.h>
#include<time.h>

# define N 10000

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block idx: %d\n block dim: %d\n thread idx: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    if(tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{   
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    clock_t d_begin = clock();
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<500,20>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t d_end = clock();

    // for(int i = 0; i < N; i++)
    // {
    //     printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    // }

    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    clock_t h_begin = clock();
    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_a[i] + h_b[i]);
    }
    clock_t h_end = clock();

    printf("Time taken by GPU: %f\n", (double)(d_end - d_begin) / CLOCKS_PER_SEC);
    printf("Time taken by CPU: %f\n", (double)(h_end - h_begin) / CLOCKS_PER_SEC);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

