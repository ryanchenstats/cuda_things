# include <stdio.h>

__device__ void Device1()
{
    printf("Device 1");
}

__device__ void Device2() 
{
    printf("Device 2");
}

__global__ void bitch()
{
    Device1();
    Device2();
}

void sub_Function_in_Host()
{
    bitch<<<1, 10>>>();
    cudaDeviceSynchronize();
}

int main()
{
    sub_Function_in_Host();

    return 0;
}
