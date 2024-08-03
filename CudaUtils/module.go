
package cutils

/*

#cgo LDFLAGS: -Wl,-rpath=${SRCDIR}/lib -L${SRCDIR}/lib -lmatmul -lpmatmul -ltranspose -lsum -lsubstract -ladd -lvecmatsum

#include <stdlib.h>


void VectorMatrixSumWrapper(double* out, const double* matrix, const double* vector, int rows, int cols); //done
void AddWrapper(double* out, const double* a, const double* b, int size);  // done
void SubtractWrapper(double* out, const double* a, const double* b, int size); // done
void SumWrapper(double* out, const double* in, int size); //done
void TransposeWrapper(double* out, const double* in, int rows, int cols); //done
void PMatMulWrapper(double* C, const double* A, const double* B, int M, int N); //done
void MatMulWrapper(double* C, const double* A, const double* B, int M, int N, int K); //done
*/
import "C"
import (
    "unsafe"
    "fmt"
    //"os/exec"

    
)



func matrixMul(C, A, B [][]float64, M, N, K int) {
    flatC := make([]float64, M*K)
    flatA := flatten(A, M, N)
    flatB := flatten(B, N, K)

    cC := (*C.double)(unsafe.Pointer(&flatC[0]))
    cA := (*C.double)(unsafe.Pointer(&flatA[0]))
    cB := (*C.double)(unsafe.Pointer(&flatB[0]))
    cM := C.int(M)
    cN := C.int(N)
    cK := C.int(K)

    C.MatMulWrapper(cC, cA, cB, cM, cN, cK)

    for i := 0; i < M; i++ {
        C[i] = flatC[i*K : (i+1)*K]
    }
}

func pMatrixMul(C, A, B [][]float64, M, N int) {
    flatC := make([]float64, M*N)
    flatA := flatten(A, M, N)
    flatB := flatten(B, M, N)

    cC := (*C.double)(unsafe.Pointer(&flatC[0]))
    cA := (*C.double)(unsafe.Pointer(&flatA[0]))
    cB := (*C.double)(unsafe.Pointer(&flatB[0]))
    cM := C.int(M)
    cN := C.int(N)

    C.PMatMulWrapper(cC, cA, cB, cM, cN)

    for i := 0; i < M; i++ {
        C[i] = flatC[i*N : (i+1)*N]
    }
}



func flatten(matrix [][]float64, rows, cols int) []float64 {
    flat := make([]float64, 0, rows*cols)
    for _, row := range matrix {
        flat = append(flat, row...)
    }
    return flat
}

func MatMul(A,B *[][]float64) ([][]float64,error) {
  M,N1 := Shape(A)
  N2,K := Shape(B)

  if N2 != N1 {return nil,fmt.Errorf("the number of columns in the first matrix has to be equal to the number of rows in the second matrix")}

  C := make([][]float64,M)
  matrixMul(C,(*A),(*B),M,N1,K)

  return C,nil
}

func PMatMul(A,B *[][]float64) ([][]float64,error) {
    
  M1,N1 := Shape(A)
  M2,N2 := Shape(B)

  if M2 != M1 || N2 != N1 {return nil,fmt.Errorf("matrix dimensions must be equal to each other")}

  C := make([][]float64,M1)

  pMatrixMul(C,(*A),(*B),M1,N1)
  return C,nil


}

func Transpose(matrix [][]float64) [][]float64 {
    rows, cols := Shape(&matrix)
    transposed := make([][]float64, cols)
    flatMatrix := flatten(matrix, rows, cols)
    flatTransposed := make([]float64, rows*cols)
    C.TransposeWrapper((*C.double)(&flatTransposed[0]), (*C.double)(&flatMatrix[0]), C.int(rows), C.int(cols))

    for i := 0; i < cols; i++ {
        transposed[i] = make([]float64, rows)
        for j := 0; j < rows; j++ {
            transposed[i][j] = flatTransposed[i*rows+j]
        }
    }

    return transposed
}
func VectorMatrixSum(A,vector [][]float64) [][]float64 {
  M,N := Shape(&A)
  vM,vN := Shape(&vector)
  flatC := make([]float64, M*N)
  flatA := flatten(A, M, N)
  flatV := flatten(vector, vM, vN)

  cC := (*C.double)(unsafe.Pointer(&flatC[0]))
  cA := (*C.double)(unsafe.Pointer(&flatA[0]))
  cV := (*C.double)(unsafe.Pointer(&flatV[0]))
  cM := C.int(M)
  cN := C.int(N)

  C.VectorMatrixSumWrapper(cC, cA, cV, cM, cN)
  
  C := make([][]float64,M)
  for i := 0; i < M; i++ {
      C[i] = flatC[i*N : (i+1)*N]
  }  
  return C 
}

func Sum(Matrix *[][]float64) [][]float64 {
  
  var C float64;
  M,N := Shape(Matrix);
  flatA := flatten((*Matrix),M,N);
  
  cC := (*C.double)(unsafe.Pointer(&C));
  cA := (*C.double)(unsafe.Pointer(&flatA[0]));
  cM := C.int(M * N);

  C.SumWrapper(cC,cA,cM);
  
  val := make([][]float64,1);
  val[0] = make([]float64,1);

  val[0][0] = C;

  return val;

}


func Subtract(A, B *[][]float64) [][]float64 {
  M,N := Shape(A);
  m1,n1 := Shape(B);
  if M != m1 || N != n1 {panic("matrix shapes must be equal to each other")}
  
  flatC := make([]float64,M*N);
  flatA := flatten((*A),M,N);
  flatB := flatten((*B),M,N);
  
  cC := (*C.double)(unsafe.Pointer(&flatC[0]));
  cA := (*C.double)(unsafe.Pointer(&flatA[0]));
  cB := (*C.double)(unsafe.Pointer(&flatB[0]));
  cM := C.int(M*N);

  C.SubtractWrapper(cC,cA,cB,cM);
  C := make([][]float64,M);
  for i := 0; i < M; i++ {
     C[i] = flatC[i*N : (i+1)*N]
  }  

  return C;

}

func Add(A, B *[][]float64) [][]float64 {
  M,N := Shape(A);
  m1,n1 := Shape(B);
  if M != m1 || N != n1 {panic("matrix shapes must be equal to each other")}
  
  flatC := make([]float64,M*N);
  flatA := flatten((*A),M,N);
  flatB := flatten((*B),M,N);
  
  cC := (*C.double)(unsafe.Pointer(&flatC[0]));
  cA := (*C.double)(unsafe.Pointer(&flatA[0]));
  cB := (*C.double)(unsafe.Pointer(&flatB[0]));
  cM := C.int(M*N);

  C.AddWrapper(cC,cA,cB,cM);
  C := make([][]float64,M);
  for i := 0; i < M; i++ {
     C[i] = flatC[i*N : (i+1)*N]
  }  

  return C;

}

func Shape(matrix *[][]float64) (int,int) {
    
  return len(*matrix),len((*matrix)[0])

}

