import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from linear_model import Model

class Plotter():
  def __init__(self, data_X, data_Y):
    self.data_X = data_X
    self.data_Y = data_Y

  def plot_fit(self, x, y):
    plt.figure()
    plt.scatter(self.data_X.numpy(), self.data_Y.numpy())
    plt.plot(x,y,'r')
    plt.ylabel('y')
    plt.xlabel('x')
  
  def plot_loss(self, losses):
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')

  def show(self):
    plt.show()

def main():
  model = Model() 

  e = torch.randint(-8,9,(50,1), dtype=torch.float)
  X = torch.linspace(1,50,50).reshape(-1,1)
  Y = 2*X + 1 + e
  x_model = np.array([X.min(), X.max()])

  plotter = Plotter(X, Y)

  w1,b1 = model.getWeight(), model.getBias()
  y1  = w1*x_model + b1
  plotter.plot_fit(x_model, y1)

  losses = model.linear_regression(X, Y)

  plotter.plot_loss(losses)

  wf,bf = model.getWeight(), model.getBias()
  yf = wf*x_model+bf
  plotter.plot_fit(x_model, yf)

  plotter.show()

if __name__=="__main__":
  main()
