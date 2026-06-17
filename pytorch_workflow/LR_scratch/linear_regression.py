import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#generating data using linear regression formula

#creating unknowm weights and bias
weight = 0.7
bia =0.3
start = 0
end =1
step =0.02
x= torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * x + bia
#print(len(x),len(y))

#spliting data into train and test data_set
train_split = int(0.8 *len(x))
x_train,y_train = x[:train_split],y[:train_split]
x_test,y_test = x[train_split:],y[train_split:]
#print(x_train,y_train,x_test,y_test)

#visualize train and test data_set

def visualize_prediction(train_data = x_train,
                         train_labels =y_train,
                         test_data = x_test,
                         test_labels = y_test,
                         prediction =None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label = "training data")
    plt.scatter(test_data,test_labels,c="g",s=4,label = "testing data")

    if prediction is not None:
        plt.scatter(test_data,prediction,c="r",s=4,label = "prediction")

    plt.legend(prop={"size":14})
    plt.show()       

#visualize_prediction()    

#simple linear regression model

class linear_regression_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights= nn.Parameter(torch.randn(1,
                                               requires_grad=True))
        
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True))
        
    def forward(self, x: torch.tensor):
        return self.weights * x +self.bias

"""torch.manual_seed(42)#generate same value
model_0 = linear_regression_model()
list =list(model_0.state_dict()) 
print(list)"""
torch.manual_seed(42)#generate same value - removed to get different results each run

model_0 = linear_regression_model()
#print("Initial weights:", model_0.state_dict())

with torch.inference_mode():# inference mode disables or does not track gradient/require_grad so torch does not keep track
    y_perd = model_0(x_test)

#visualize_prediction(prediction=y_perd)

#training model
## the whole idea of training data is to moving unknown parameters to some known parameters
## or poor representation of data to better representation of data
## to measure how poor the model is we us loss function(or may be also called cost function,depending on case)
#loss function: it measures how wrong our model is to our output prediction
#optimizer: take in account of loss function and adjust model parameters (weights and bias) to improve loss function we need two things:
'''#trainig loop;
#testing loop'''

#setup loss function(a.k.a l1\mae)
loss_function = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

"""#trainig loop ( and testing loop)
# 1. moving through data
# 2. forward pass (moving data through forward function) to make prediction.(froward propagation)
# 3.calculate the loss
# 4.optimizer zero grad
# 5.loss backward to move backward through the network to calculate gradient of each parameter of our model with respect to loss
# 6.optimizer step - to use optimizer to adjust our model parameter to try and improve loss(gradient descent)
# optimizer has two parts:
#params
#learning rate
#epoch is one loop through data"""
#print(model_0.state_dict())
"""
epochs = 200
epoch_count =[]
loss_value =[]
test_loss_value =[]

for epoch in range(epochs):

    model_0.train()
    #forward pass
    y_pred = model_0(x_train)
    #loss function
    loss = loss_function(y_pred,y_train)
    #optimize zero_grad
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    #gradient descent
    optimizer.step()

    model_0.eval()
    #print("\nFinal weights:", model_0.state_dict())
    #if epoch % 2 == 0:
        #print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_function(test_pred,y_test)

    if epoch % 10 ==0:
        epoch_count.append(epoch)
        loss_value.append(loss)
        test_loss_value.append(test_loss)
        #print(f"eproch:{epoch},loss:{loss},test loss:{test_loss}")"""

"""plt.plot(epoch_count, np.array([l.detach().cpu().numpy() for l in loss_value]), label='train loss')
plt.plot(epoch_count, np.array([l.detach().cpu().numpy() for l in test_loss_value]), label='test loss')
plt.title("training and test loss curves")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()"""
#print(model_0.state_dict())
#visualize_prediction(prediction=test_pred) 
#print(model_0.state_dict()) 
  
#saving and loading model state dict
from pathlib import Path

#creating model directory
model_path = Path(__file__).parent  # Points to the data folder
#creating model save path
model_name = "model_0_workflow.pth"
model_save_path = model_path/model_name
#save model
#torch.save(obj=model_0.state_dict(),f=model_save_path)

#load model
loaded_model_0 = linear_regression_model()
loaded_model_0.load_state_dict(torch.load(f=model_save_path))
print("loaded model state dict:", loaded_model_0.state_dict())

#make prediction with loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_pred = loaded_model_0(x_test)

visualize_prediction(prediction=loaded_model_pred)