extern "C" {

__global__ void PMatMulKernel(double* C, const double* A, const double* B, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        int index = i * N + j;
        C[index] = A[index] * B[index];
    }
}

void PMatMulWrapper(double* C, const double* A, const double* B, int M, int N) {
    double *d_A, *d_B, *d_C;
    size_t size = M * N * sizeof(double);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    PMatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, M, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}

