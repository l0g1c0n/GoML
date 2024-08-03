extern "C" {

__global__ void VectorMatrixSumKernel(double* out, const double* matrix, const double* vector, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        int index = i * cols + j;
        out[index] = matrix[index] + vector[j];
    }
}

void VectorMatrixSumWrapper(double* out, const double* matrix, const double* vector, int rows, int cols) {
    double *d_matrix, *d_vector, *d_out;
    size_t matrixSize = rows * cols * sizeof(double);
    size_t vectorSize = cols * sizeof(double);

    cudaMalloc(&d_matrix, matrixSize);
    cudaMalloc(&d_vector, vectorSize);
    cudaMalloc(&d_out, matrixSize);

    cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, vectorSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    VectorMatrixSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_matrix, d_vector, rows, cols);

    cudaMemcpy(out, d_out, matrixSize, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_out);
}

}

