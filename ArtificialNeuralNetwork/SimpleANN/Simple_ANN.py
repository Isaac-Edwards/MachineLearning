import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ANN_model import Model
from sklearn.model_selection import train_test_split

df = pd.read_csv('~/MachineLearning/UdemyData/iris.csv') # is there a way to make this more platform-independent?

model = Model()

# training

X = df.drop('target', axis=1)
Y = df['target']
X = X.values
Y = Y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
  y_pred = model.forward(X_train)
  loss = criterion(y_pred, Y_train)
  losses.append(loss)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# testing

correct = 0

with torch.no_grad():
  for i, data in enumerate(X_test):
    y_eval = model.forward(data)
    if y_eval.argmax().item()==Y_test[i]:
      correct+=1

print(f'We got {correct} of {len(X_test)} correct!')
