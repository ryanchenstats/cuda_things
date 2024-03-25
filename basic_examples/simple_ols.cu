// simple OLS with cuda for beta calculation
// no matrix operations, just loops and vectorization

#include<iostream>
#include<cuda_runtime.h>
#include<time.h>

# define N 10000
# define maxRandInt 100
# define beta 5.3f

// box muller
__host__ float sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

// take diff squares
__global__ void square_diff_mean(float *ss, float *v1, float *v2, float *v1mean, float *v2mean)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        ss[idx] = (v1[idx] - v1mean[idx]) * (v2[idx] - v2mean[idx]);
    }
}

// sum only 
__host__ void sum(float *h_sum, float *v)
{
    float csum = 0.0;
    for (int i = 0; i < N; i++)
    {
        csum += v[i];
    }
    *h_sum = csum;
}

__host__ void mean(float *mean_result, float *v)
{
    float sum = 0.0;
    float mean = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum += v[i];
    }
    mean = sum / N;
    for (int i = 0; i < N; i++)
    {
        mean_result[i] = mean;
    }
}
float host_beta(float *x1, float *y, float *xmean, float *ymean)
{
    float hh_ssxy = 0.0;
    float hh_ssx = 0.0;
    for (int i = 0; i < N; i++)
    {
        hh_ssxy += (x1[i] - xmean[i]) * (y[i] - ymean[i]);
        hh_ssx += (x1[i] - xmean[i]) * (x1[i] - xmean[i]);
    }
    return hh_ssxy / hh_ssx;
}

int main()
{   
    float *x1, *y, *intercept, *rbeta, *ymean, *xmean;
    float *sxy, *sx;
    float *ssxy, *ssx;
    float *d_x1, *d_y, *d_intercept, *d_rbeta, *d_ymean, *d_xmean;
    float *d_sx, *d_sxy;
    // allocate local memory
    intercept = (float *) malloc(N * sizeof(float));
    x1 = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));
    rbeta = (float *) malloc(N * sizeof(float));
    xmean = (float *) malloc(N * sizeof(float));
    ymean = (float *) malloc(N * sizeof(float));
    sxy = (float *) malloc(N * sizeof(float));
    sx = (float *) malloc(N * sizeof(float));
    ssxy = (float *) malloc(sizeof(float));
    ssx = (float *) malloc(sizeof(float));

    // initialize data
    srand ( time(NULL) );
    for (int i = 0; i < N; i++)
    {
        intercept[i] = 1.0;
        x1[i] = maxRandInt * (float)rand()/(float)(RAND_MAX);
        y[i] =  beta * x1[i] + intercept[i];//+ sampleNormal();
    }

    // get means
    mean(xmean, x1);
    mean(ymean, y);

    clock_t d_begin = clock();
    // allocate memory on device
    cudaMalloc((void**)&d_intercept, N * sizeof(float));
    cudaMalloc((void**)&d_x1, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_rbeta, sizeof(float));
    cudaMalloc((void**)&d_xmean, N * sizeof(float));
    cudaMalloc((void**)&d_ymean, N * sizeof(float));
    cudaMalloc((void**)&d_sxy, N * sizeof(float));
    cudaMalloc((void**)&d_sx, N * sizeof(float));

    // copy data to device
    cudaMemcpy(d_intercept, intercept, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1, x1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rbeta, rbeta, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xmean, xmean, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ymean, ymean, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sxy, sxy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx, sx, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // call regression
    int numBlocks = N / 1000;
    int threadsPerBlock = 1000;
    square_diff_mean<<<numBlocks,threadsPerBlock>>>(d_sxy, d_y, d_x1, d_ymean, d_xmean); 
    square_diff_mean<<<numBlocks,threadsPerBlock>>>(d_sx, d_x1, d_x1, d_xmean, d_xmean);
    // cudaDeviceSynchronize();


    cudaMemcpy(sxy, d_sxy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sx, d_sx, N * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t d_end = clock();
    printf("Time taken (device): %f\n", (double)(d_end - d_begin) / CLOCKS_PER_SEC);

    sum(ssxy, sxy);
    sum(ssx, sx);

    float hat_beta, hat_intercept;
    hat_beta = *ssxy / *ssx;
    hat_intercept = *ymean - hat_beta * *xmean;

    printf("beta: %f\n", hat_beta);
    printf("intercept: %f\n", hat_intercept);

    clock_t h_begin = clock();
    float hh_beta;
    hh_beta = host_beta(x1, y, xmean, ymean);
    clock_t h_end = clock();
    printf("Time taken (host): %f\n", (double)(h_end - h_begin) / CLOCKS_PER_SEC);

    // cudaMemcpy(beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("beta: %f", beta);
    // return 0;
}
