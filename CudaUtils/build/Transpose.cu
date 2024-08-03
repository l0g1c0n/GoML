extern "C" {

__global__ void TransposeKernel(double* out, const double* in, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

void TransposeWrapper(double* out, const double* in, int rows, int cols) {
    double *d_in, *d_out;
    size_t size = rows * cols * sizeof(double);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, rows, cols);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

}

