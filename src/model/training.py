from math import ceil
from pathlib import Path

import torch
from torch.autograd import Variable

from regression import LinearRegressionModel

# Declaring variables
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0], [8.0], [10.0]]))

def save_model(model):
    with open(str(Path.cwd()) +'/src/model/model.pkl', 'wb') as file:
        torch.save(model, file)

# Create model
model = LinearRegressionModel()

# Trainning model
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for time in range(50):
    pred_y = model(x_data)

    loss = criterion(pred_y, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Scenario {}, loss {}'.format(time, loss.item()))


new_var = Variable(torch.Tensor([[6.0]]))
pred_y = model(new_var)
print("Prediction (after training)", 6, ceil(pred_y.item()))

save_model(model)
