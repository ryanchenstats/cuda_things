// wip

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define N 2
#define M 2
#define maxRandInt 100

__global__ void multiply(float **A, float **B, float **C)
{
    int x_step = blockDim.x * blockIdx.x + threadIdx.x;
    int y_step = blockDim.y * blockIdx.y + threadIdx.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for (int i = x_step; i < N; i += stridex)
    {
        for (int j = y_step; j < N; j += stridey)
        {
            for (int k = 0; k < M; k += 1)
            {
                C[i][j] += A[i][k] * B[k][j];
                printf("%d", C[i][j]);
            }
        }
    }
}

int main()
{
    float **A = new float *[N];
    float **B = new float *[M];
    float **C = new float *[N];
    float **D = new float *[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new float[M];
    }
    for (int i = 0; i < M; i++)
    {
        B[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        C[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            A[i][j] = maxRandInt * (float)rand() / (float)(RAND_MAX);
            B[j][i] = maxRandInt * (float)rand() / (float)(RAND_MAX);
        }
        for (int k = 0; k < N; k++)
        {
            C[i][k] = 0.0;
            D[i][k] = 0.0;
        }
    }

    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%f ", C[i][j]);
    //     }
    //     cout << endl;
    // }

    cudaMallocManaged(&A, N * M * sizeof(float));
    cudaMallocManaged(&B, M * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));

    multiply<<<N, M>>>(A, B, C);
    cudaDeviceSynchronize();

    return 0;
}