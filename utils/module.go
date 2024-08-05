package utils

import (
  "math"
  "math/rand"
)




func XavierWeightInit(inFeatures, outFeatures int) [][]float64 {
  limit := math.Sqrt(6.0 / (float64(inFeatures + outFeatures)))
  weights := make([][]float64, inFeatures)
  for i := range weights {
    weights[i] = make([]float64, outFeatures)
    for j := range weights[i] {
      weights[i][j] = rand.Float64()*2*limit - limit
    }
  }
  return weights
}

func LecunBiasInit(Features int) [][]float64 {
  limit := math.Sqrt(1.0 / float64(Features))
  biases := make([]float64, Features)
  unsqueezed := make([][]float64,1)

  for i := range biases {
    biases[i] = rand.Float64()*2*limit - limit
  }
  unsqueezed[0] = biases

  return unsqueezed
}

func MatMul(m1 *[][]float64, m2 *[][]float64) [][]float64 {
  if len((*m1)[0]) != len(*m2) {
    panic("the number of rows in the first matrix is not equal to the numbers of columns in the second matrix")
  }
  desiredShape := []int{len(*m1), len((*m2)[0])}
  finalMatrix := make([][]float64, desiredShape[0])
  for i := range desiredShape[0] {
    row := make([]float64, desiredShape[1])
    for k := range desiredShape[1] {
      var out float64 = 0
      for j := range len(*m2) { // Iterate over columns in m2
        out += (*m1)[i][j] * (*m2)[j][k]
      }
      row[k] = out
    }
    finalMatrix[i] = row
  }
  return finalMatrix
}

func PMatMul (m1 *[][]float64,m2 *[][]float64) [][]float64 {
  matrix_shape := Shape(m1) 
  new_matrix := make([][]float64,matrix_shape[0])
  for i := range matrix_shape[0] {
    row := make([]float64,matrix_shape[1])
    for j := range matrix_shape[1] {

      row[j] = (*m1)[i][j] * (*m2)[i][j]
    }
    new_matrix[i] = row

  } 
  return new_matrix
}

func Shape(matrix *[][]float64) [2]int {
    
  return [2]int{len(*matrix),len((*matrix)[0])}

}

func Transpose(matrix [][]float64) [][]float64 {
  rows := len(matrix)
  if rows == 0 {
    return nil
  }
  cols := len(matrix[0])
  transposed := make([][]float64, cols)
  for i := range transposed {
    transposed[i] = make([]float64, rows)
    for j := range matrix {
      transposed[i][j] = matrix[j][i]
    }
  }
  return transposed
}

func Subtract(matrix1, matrix2 *[][]float64) [][]float64 {
  if len(*matrix1) != len(*matrix2) || len((*matrix1)[0]) != len((*matrix2)[0]) {
    panic("must have same dims")
  }

  rows := len(*matrix1)
  cols := len((*matrix1)[0])
  result := make([][]float64, rows)
  for i := range result {
    result[i] = make([]float64, cols)
    for j := range (*matrix1)[i] {
      result[i][j] = (*matrix1)[i][j] - (*matrix2)[i][j]
    }
  }
  return result
}

func MatOp(Matrix *[][]float64,f func(float64) float64) [][]float64 {
  matrix_shape := Shape(Matrix)
  var new_matrix [][]float64 = make([][]float64,matrix_shape[0])

  for i := range matrix_shape[0] {
    row := make([]float64,matrix_shape[1])
    for j := range matrix_shape[1] {
      
      row[j] = f((*Matrix)[i][j])
    
    }
  new_matrix[i] = row
  }
  return new_matrix

}


func Add(matrix1, matrix2 *[][]float64) [][]float64 {
  if len(*matrix1) != len(*matrix2) || len((*matrix1)[0]) != len((*matrix2)[0]) {
    panic("must have same dims")
  }

  rows := len(*matrix1)
  cols := len((*matrix1)[0])
  result := make([][]float64, rows)
  for i := range result {
    result[i] = make([]float64, cols)
    for j := range (*matrix1)[i] {
      result[i][j] = (*matrix1)[i][j] + (*matrix2)[i][j]
    }
  }
  return result
}
func Sum(Matrix *[][]float64) [][]float64 {
  matrix_shape := Shape(Matrix)
  val := make([][]float64,1)
  val[0] = make([]float64,1)
  for i := range matrix_shape[0] {
    for j := range matrix_shape[1] {
      val[0][0] += (*Matrix)[i][j]

    }  
  }
  return val
}


func VectorMatrixSum(m [][]float64,vector [][]float64) [][]float64 {
  m_shape := Shape(&m)
  for i := range m_shape[0] {
    unsqueezed := [][]float64{m[i]}
    m[i] = Add(&unsqueezed,&vector)[0]
    
  }
  return m 
}

func ReLU(x *[][]float64) {
  
  (*x) = MatOp(x,func(y float64)float64{if y>0{return y}else {return 0}})

}

func Sigmoid(x *[][]float64) {
  (*x) = MatOp(x,func(y float64)float64{return 1/(1+math.Exp(-y))})
}

func Tanh(x *[][]float64) {
  (*x) = MatOp(x, func(y float64) float64 {return math.Tanh(y)})
}

func LeakyReLU(x *[][]float64, alpha float64) {
  (*x) = MatOp(x, func(y float64) float64 {if y > 0 {return y} else {return alpha * y}})
}

func Softmax(x *[][]float64) [][]float64 {
  rows := len(*x)
  cols := len((*x)[0])


  expMatrix := make([][]float64, rows)
  for i := range *x {
    expMatrix[i] = make([]float64, cols)
    for j := range (*x)[i] {
      expMatrix[i][j] = math.Exp((*x)[i][j])
    }
  }
  rowSums := make([]float64, rows)
  for i := range expMatrix {
    sum := 0.0
    for _, value := range expMatrix[i] {
      sum += value
    }
    rowSums[i] = sum
  }
  for i := range expMatrix {
    for j := range expMatrix[i] {
      expMatrix[i][j] /= rowSums[i]
    }
  }
  return expMatrix
}

func Max(matrix [][]float64) (float64, int, int) {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return math.NaN(), -1, -1
    }

    maxVal := math.Inf(-1) 
    maxRow := -1
    maxCol := -1

    for i, row := range matrix {
        for j, value := range row {
            if value > maxVal {
                maxVal = value
                maxRow = i
                maxCol = j
            }
        }
    }

    return maxVal, maxRow, maxCol
}


func Min(matrix [][]float64) (float64, int, int) {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return math.NaN(), -1, -1
    }

    minVal := math.Inf(1) 
    minRow := -1
    minCol := -1

    for i, row := range matrix {
        for j, value := range row {
            if value < minVal {
                minVal = value
                minRow = i
                minCol = j
            }
        }
    }

    return minVal, minRow, minCol
}

