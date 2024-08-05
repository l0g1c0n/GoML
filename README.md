# GoML

GoML is a Go library for machine learning that provides basic tools for building and training neural networks. It includes implementations for linear layers, mean squared error loss, and momentum descent optimization.

## Installation

To use GoML in your project, you can install it using Go modules. Run the following command:

```sh
go get github.com/l0g1c0n/GoML
```

## Example Usage

Here's a simple example demonstrating how to use GoML to create and train a neural network.

### Code Example

```go
package main

import (
    "fmt"
    "github.com/l0g1c0n/GoML/cuda/nn"  // Use this import for CUDA support
    // "github.com/l0g1c0n/GoML/nn"    // Use this import for CPU support

    "github.com/l0g1c0n/GoML/utils" // import utils for activation functions and matrix operations
)

type network_struct struct {
    fc1  *nn.Linear_struct
    fc2  *nn.Linear_struct
    fc3  *nn.Linear_struct
    fc4  *nn.Linear_struct
    pass *nn.ForwardPass
}

func (self *network_struct) Forward(x [][]float64) [][]float64 {
  
  x = self.fc1.Forward(x,self.pass)
  utils.ReLU(&x)
  x = self.fc2.Forward(x,self.pass)
  utils.ReLU(&x)
  x = self.fc3.Forward(x,self.pass)
  utils.LeakyReLU(&x,0.1)
  x = self.fc4.Forward(x,self.pass)
  return x
}

func Network(input_size, hidden_size, output_size int) network_struct {
    return network_struct{
        nn.Linear(input_size, hidden_size),
        nn.Linear(hidden_size, hidden_size*2),
        nn.Linear(hidden_size*2, hidden_size),
        nn.Linear(hidden_size, output_size),
        &nn.ForwardPass{},
    }
}

func batchdata(A [][]float64, size int) [][]float64 {
    for _ = range (size - 1) {
        A = append(A, A[0])
    }
    return A
}

func main() {
    Net := Network(5, 2048, 5)
    mse := nn.MSELoss{}
    optim := nn.MomentumDescent{}
    optim.Init(Net.pass, 0.001, 0.09)

    matrix := [][]float64{{0.1, 0.2, 0.3, 0.4, 0.5}}
    target := [][]float64{{0.5, 0.4, 0.3, 0.2, 0.5}}
    BatchSize := 100

    matrix = batchdata(matrix, BatchSize)
    target = batchdata(target, BatchSize)

    for i := 0; i < 10; i++ {
        output := Net.Forward(matrix)
        error := mse.Loss(&output, &target)
        optim.Step(&mse.Gradients)

        fmt.Println("epoch: ", i, " loss: ", error[0][0])
    }
}
```

## Core Components

### Neural Network Module (`nn`)

#### `Linear_struct`

- **`Forward(x [][]float64, pass *ForwardPass) [][]float64`**
  - Computes the forward pass through a linear layer.

- **`Backward(Gradients *[][]float64, pass *ForwardPass)`**
  - Computes the backward pass to update gradients.

#### `Linear(InFeatures int, OutFeatures int) *Linear_struct`

- **`Linear`**
  - Creates a new `Linear_struct` with initialized weights and biases.

#### `MSELoss`

- **`Loss(Ypred *[][]float64, Tpred *[][]float64) [][]float64`**
  - Computes the Mean Squared Error (MSE) loss between predicted and target values.

#### `MomentumDescent`

- **`Init(pass *ForwardPass, lr float64, alpha float64)`**
  - Initializes the optimizer with learning rate and momentum.

- **`Step(Gradients *[][]float64)`**
  - Updates weights and biases based on gradients using momentum.

## CPU vs. CUDA Support

To use GoML with different computation backends, you can choose between the CPU and CUDA versions:

- **For CPU support:** Import the CPU version of the `nn` module:
  ```go
  import "github.com/l0g1c0n/GoML/nn"
  ```

- **For CUDA support:** Import the CUDA version of the `nn` module:
  ```go
  import "github.com/l0g1c0n/GoML/cuda/nn"
  ```

Make sure to have the appropriate environment set up for CUDA if you choose the CUDA version.

## Utility Functions

GoML provides a set of utility functions for matrix manipulation through the `utils` package. Below are the key functions:

- **`XavierWeightInit(inFeatures, outFeatures int) [][]float64`**
  - Initializes weights using Xavier initialization.

- **`LecunBiasInit(Features int) [][]float64`**
  - Initializes biases using LeCun initialization.

- **`MatMul(m1 *[][]float64, m2 *[][]float64) [][]float64`**
  - Performs matrix multiplication between two matrices.

- **`PMatMul(m1 *[][]float64, m2 *[][]float64) [][]float64`**
  - Performs element-wise matrix multiplication.

- **`Shape(matrix *[][]float64) [2]int`**
  - Returns the shape of the matrix as a `[2]int` array.

- **`Transpose(matrix [][]float64) [][]float64`**
  - Computes the transpose of a matrix.

- **`Subtract(matrix1, matrix2 *[][]float64) [][]float64`**
  - Subtracts one matrix from another.

- **`Add(matrix1, matrix2 *[][]float64) [][]float64`**
  - Adds two matrices.

- **`Sum(Matrix *[][]float64) [][]float64`**
  - Computes the sum of all elements in the matrix.

- **`VectorMatrixSum(m [][]float64, vector [][]float64) [][]float64`**
  - Adds a vector to each row of a matrix.

## License

MIT License
