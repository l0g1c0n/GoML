// Sum.cu
extern "C" {

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void SumKernel(double* out, const double* in, int size) {
    __shared__ double sharedSum[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sharedSum[tid] = (i < size) ? in[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddDouble(out, sharedSum[0]);
    }
}

void SumWrapper(double* out, const double* in, int size) {
    double *d_in, *d_out;
    size_t dataSize = size * sizeof(double);

    cudaMalloc(&d_in, dataSize);
    cudaMalloc(&d_out, sizeof(double));

    cudaMemcpy(d_in, in, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    SumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, size);

    cudaMemcpy(out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

}
