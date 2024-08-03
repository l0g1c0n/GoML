
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(double* C, const double* A, const double* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        double value = 0;
        for (int e = 0; e < N; ++e)
            value += A[row * N + e] * B[e * K + col];
        C[row * K + col] = value;
    }
}

extern "C" void MatrixMul(double* C, const double* A, const double* B, int M, int N, int K) {
    double *d_A, *d_B, *d_C;
    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * K * sizeof(double);
    size_t size_C = M * K * sizeof(double);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

