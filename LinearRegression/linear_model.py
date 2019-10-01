import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def getWeight(self):
      return self.linear.weight.item()

    def getBias(self):
      return self.linear.bias.item()

    def linear_regression(self, X, Y, epochs=20):
      losses = []

      for i in range(epochs):
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, Y)
        losses.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      return losses

