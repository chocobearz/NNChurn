import pandas as pd
import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=10000)

def sigmoid(x):
  return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
  return x * (1.0 - x)

#neural network class
#use 4 neurons
class NeuralNetwork:
  def __init__(self, x, y):
    self.input      = x
    self.weights1   = np.random.rand(self.input.shape[1],7043) 
    self.weights2   = np.random.rand(7043,1)                  
    self.y          = y
    self.output     = np.zeros(y.shape)

  #feed forward
  #bias is 0 or we would have + bias after dot product
  def feedforward(self):
    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    self.output = sigmoid(np.dot(self.layer1, self.weights2))


  def backprop(self):
    # chain rule for derivative of the loss function (SSE) wrt weights2 and weights1
    d_weights2 = np.dot(
      self.layer1.T, (
        2*(
          self.y - self.output
        ) * sigmoid_derivative(
          self.output
        )
      )
    )
    d_weights1 = np.dot(
      self.input.T, (
        np.dot(
          2*(
            self.y - self.output
          ) * sigmoid_derivative(
            self.output
          ), self.weights2.T
        ) * sigmoid_derivative(
          self.layer1
        )
      )
    )
    # update the weights with the derivative (slope) of the loss function
    self.weights1 += d_weights1
    self.weights2 += d_weights2

cd = pd.read_csv("oneHotNNData.csv")
churn = cd["churn"]
regressors = cd.loc[:, cd.columns != 'churn']
churn = churn.to_numpy()
cd = cd.to_numpy()
churn = churn.reshape(7043, 1)

nn = NeuralNetwork(regressors, churn)

for i in range(10):
  nn.feedforward()
  nn.backprop()

print(nn.output)
print("done")