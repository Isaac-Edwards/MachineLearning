import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class Plotter():
  def __init__(self, data_X, data_Y):
    self.data_X = data_X
    self.data_Y = data_Y

  def plot(self, x, y):
    plt.scatter(self.data_X.numpy(), self.data_Y.numpy())
    plt.plot(x,y,'r')
    plt.ylabel('y')
    plt.xlabel('x')

  def show(self):
    plt.show()

def main():
  torch.manual_seed(59)
  model = Model()

  e = torch.randint(-8,9,(50,1), dtype=torch.float)
  X = torch.linspace(1,50,50).reshape(-1,1)
  Y = 2*X + 1 + e

  plotter = Plotter(X, Y)

  x1 = np.array([X.min(), X.max()])
  w1,b1 = model.linear.weight.item(), model.linear.bias.item()
  y1  = w1*x1 + b1

  plotter.plot(x1,y1)
  plotter.show()

if __name__=="__main__":
  main()
