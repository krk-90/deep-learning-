import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

#Data(creating data)
weight =0.7
bias = 0.3
start = 0
end =1
step = 0.02
x = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * x + bias

#split data
train_test = int(0.8 * len(x))
x_train,y_train = x[:train_test],y[:train_test]
x_test,y_test = x[train_test:],y[train_test:]

#visualize train and test
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

#model(linear regression using torch.nn sub-models)    

class linear_regression_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x :torch.tensor):
        return self.linear_layer(x)
    
torch.manual_seed(42)    
linear_model = linear_regression_model()

#loss function
loss_function = nn.L1Loss()
#optimizer
optimizer = torch.optim.SGD(params=linear_model.parameters(),lr=0.01)

#training loop
torch.manual_seed(42)

epochs =200
for epoch in range(epochs):
    linear_model.train()

    y_pred = linear_model(x_train)

    loss = loss_function(y_pred,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    linear_model.eval()
    with torch.inference_mode():
        test_pred = linear_model(x_test)
        test_loss = loss_function(test_pred,y_test)

    """if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
print(linear_model.state_dict())
"""
#visualizing the prediction
#visualize_prediction(prediction=test_pred)

#saving and loading model
model_path = Path(__file__).parent
model_name = "linear_regression_model.pth"
model_save_path = model_path/model_name

torch.save(obj=linear_model.state_dict(),f=model_save_path)

#loading model
loaded_model =linear_regression_model()
loaded_model.load_state_dict(torch.load(f=model_save_path))
print("loaded model state dict:", loaded_model.state_dict())

#making prediction
loaded_model.eval()
with torch.inference_mode():
    prediction = loaded_model(x_test)

visualize_prediction(prediction=prediction)    

