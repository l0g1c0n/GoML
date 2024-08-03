package nn

import (
  "time"
  "math/rand"
  "GoML/utils"
)


type ForwardPass struct {

  Layers []*Linear_struct
}
func (self *ForwardPass) Clear() {

  self.Layers = []*Linear_struct{}
}

type Linear_struct struct {

  Weight [][]float64
  Bias [][]float64
  recent_input [][]float64
  Delta_Weight [][]float64
  index int
  activation_function bool
}


func (self *Linear_struct) Forward(x [][]float64,pass *ForwardPass) [][]float64 {
  self.recent_input = x
  recent_output := utils.VectorMatrixSum(utils.MatMul(&x,&self.Weight),self.Bias)
  pass.Layers = append(pass.Layers, self)
  self.index = len(pass.Layers)-1
  return recent_output
}

func (self *Linear_struct) Backward(Gradients *[][]float64,pass *ForwardPass) {
  var num_layers int = len(pass.Layers) -1


  if self.index == num_layers {
     self.Delta_Weight = *Gradients 
  }  else {
    previous_delta := (*pass).Layers[self.index+1].Delta_Weight
    previous_weight := utils.Transpose((*pass).Layers[self.index+1].Weight)
    
    self.Delta_Weight = utils.MatMul(&previous_delta,&previous_weight)
  }
}


func Linear(InFeatures int ,OutFeatures int) *Linear_struct {
  rand.Seed(time.Now().UnixNano())
  Weights := utils.XavierWeightInit(InFeatures,OutFeatures)
  Bias := utils.LecunBiasInit(OutFeatures)
  var linear_struct Linear_struct
  linear_struct.Weight = Weights
  linear_struct.Bias = Bias
  linear_struct.activation_function = false
  return &linear_struct 
}

type MSELoss struct {
  
  Gradients [][]float64
}

func (self *MSELoss) Loss(Ypred *[][]float64, Tpred *[][]float64) [][]float64 {
    self.Gradients = utils.Subtract(Tpred,Ypred)
    
    loss := utils.MatOp(&self.Gradients,func(x float64) float64 {return (x*x)/2})
    summed_loss := utils.Sum(&loss)
    return summed_loss
    
  
}

type MomentumDescent struct {
  
  Pass *ForwardPass
  Lr float64
  Alpha float64
  p_Delta [][][]float64
}
func (self *MomentumDescent) Init(pass *ForwardPass,lr float64,alpha float64) {

  self.Pass = pass
  self.Lr = lr 
  self.Alpha = alpha
}
func (self MomentumDescent) Step(Gradients *[][]float64) {
  
  num_layers := len(self.Pass.Layers) -1
  
  self.Pass.Layers[num_layers].Backward(Gradients,self.Pass)

  AddMomentum := len(self.p_Delta) != 0
  
  for i := num_layers-1; i >=0; i-- {
    current_layer := self.Pass.Layers[i]
    current_layer.Backward(&self.Pass.Layers[i+1].Delta_Weight,self.Pass)
    transposed_recent_input := utils.Transpose(current_layer.recent_input)
    Weight_grads := utils.MatMul(&transposed_recent_input,&current_layer.Delta_Weight)
    shape := utils.Shape(&Weight_grads)
    Bias_grads := make([][]float64,shape[1])
    
    Delta := utils.MatOp(&Weight_grads,func(x float64) float64 {return self.Lr * x})

    w_grads_transposed := utils.Transpose(Delta)
    for i := range shape[1] {
      unsqueezed := make([][]float64,1)
      unsqueezed[0] = w_grads_transposed[i]
      summed_row := utils.Sum(&unsqueezed)
      Bias_grads[i] = summed_row[0]
    }
    Bias_grads = utils.Transpose(Bias_grads)




    if AddMomentum == true { 
    prev_delta := utils.MatOp(&self.p_Delta[i],func(x float64) float64 {return self.Alpha * x})
    Delta = utils.Add(&Delta,&prev_delta)
    }
    self.p_Delta = append(self.p_Delta, Delta)
    current_layer.Weight = utils.Add(&current_layer.Weight,&Delta)
    current_layer.Bias = utils.Add(&current_layer.Bias,&Bias_grads)

  }
  self.Pass.Clear()
}



