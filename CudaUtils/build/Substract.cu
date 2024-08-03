extern "C" {

__global__ void SubtractKernel(double* out, const double* a, const double* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = a[i] - b[i];
    }
}

void SubtractWrapper(double* out, const double* a, const double* b, int size) {
    double *d_a, *d_b, *d_out;
    size_t dataSize = size * sizeof(double);

    cudaMalloc(&d_a, dataSize);
    cudaMalloc(&d_b, dataSize);
    cudaMalloc(&d_out, dataSize);

    cudaMemcpy(d_a, a, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, dataSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    SubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_a, d_b, size);

    cudaMemcpy(out, d_out, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

}

