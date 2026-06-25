import torch
import numpy as np

#for predicting yeild of apples or orange
#temp ,rainfall, humidity

x = np.array([[67,73,81],
              [75,88,164],
              [99,145,77],
              [102,43,37],
              [67,87,44]])

y = np.array([[56,70],
              [81,101]
              ,[56,45]
              ,[40,38],
              [50,60]])

#converting to tensor
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

w= torch.randn(2,3,requires_grad=True)
b = torch.randn(2,requires_grad=True)

y_pred = x @ w.T + b

error =torch.nn.MSELoss()(y_pred,y)
root = torch.sqrt(error)
print(root)