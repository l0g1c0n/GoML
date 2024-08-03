
extern "C" void MatrixMul(double* C, const double* A, const double* B, int M, int N, int K);

extern "C" void MatMulWrapper(double* C, const double* A, const double* B, int M, int N, int K) {
    MatrixMul(C, A, B, M, N, K);
}


