import numpy as np
import torch
"""
tensor = torch.tensor([1,2,3.],dtype = None,device= None,requires_grad= False)
t = tensor.shape
t0 = tensor.ndim
t1 = torch.arange(1,20,2)
t2 = torch.zeros(size=(3,4))
t3 = torch.ones(size=(3,4))
t4 = torch.zeros_like(input=t1)
t5 = torch.ones_like(input=t1)
rand = torch.rand(3,4)
rand1 = torch.rand(3,4,4)
rand_s = rand1.shape
rand_nd = rand1.ndim
float_16 = tensor.type(torch.float16)
tenso =  torch.tensor([1,2,3.],dtype=torch.long)
tens = torch.tensor([3,4,5],dtype=torch.int32)
#tensor dtype error
#tensor device error 
#tensor shape error
tens_de = tenso * tens
##print(tens_de.device)

#Manipulating tensor
#addition
#substraction
#multiplication (element-wise)
#division
#matrix multiplication


tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([4,5,6])
add = torch.add(tensor1,tensor2)
sub = torch.sub(tensor1,tensor2)
div = torch.divide(tensor1,tensor2)
element_wise_multi = torch.mul(tensor1,tensor2)
matrix_multi = torch.matmul(tensor1,tensor2)
dict ={
    add,sub,div,element_wise_multi,matrix_multi
}
for i in dict:
    #print(i)
import time
start = time.perf_counter()
value = 0 
for i in range(len(tensor1)):
    value += tensor1* tensor1
value
#print(value)    
end = time.perf_counter()
##print(end-start)

#matrix multiplication when the tensors are same shape
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
tensor4 = torch.tensor([[7,8],[9,10],[11,12]]) 
matmul = torch.matmul(tensor3,tensor4.T)"""

#tensor operation and gradient
'''x = torch.tensor(3.)
w = torch.tensor(4.,requires_grad=True)
b = torch.tensor(5.,requires_grad=True)
y = x * w + b
y.backward()#computes derivative
#print(x.grad)
#print(w.grad)
#print(b.grad)'''

#introperability with numpy
'''import numpy as np 
x = np.array([[1.,2],[3,4]])
y = torch.from_numpy(x)
z = y.numpy()
#print(z)'''

#mean ,sum,mim,max,argmax,argmin
'''
x= torch.arange(0,100,10)
min =torch.min(x)
max = torch.max(x)
mean = torch.mean(x.float())
sum = torch.sum(x)
argmax = torch.argmax(x)
argmin = torch.argmin(x)
#print(mean)'''

#reshape,viewing,stacking tensors,squeezing,unsqueezing,permute
"""x = torch.arange(1.,10)
r_s =x.reshape(9,1)
#z = x.view(1,9)
#z[:,0] = 5
#stacked_x = torch.stack([x,x,x,x])
#stacked_x = torch.stack([x,x,x,x],dim=1)
##print(stacked_x)

#x = torch.zeros(2,1,2,1,2)
#y =torch.squeeze(x)
#y =torch.squeeze(x,0)
y =torch.squeeze(r_s,1)
z = y.unsqueeze(dim=0)#or dim = 1
#print(z)
x = torch.randn(32,32,3)
x1 = x.size()
y = torch.permute(x,(2,0,1)).size()
#print(x1,y)"""
#indexing
#x = torch.arange(1,10).reshape(1,3,3)
#print(x[0,0:2])
#print(x[0,0])
#print(x[0,1,1])
#print(x[:,2])
#print(x[:,1,1])
#print(x[0,0,:])
#print(x[:,:,2])
"""
# from numpy to tensor (tensor to numpy)
array= np.arange(1.,9.)
tensor = torch.from_numpy(array)
arr = tensor.numpy()
#print(tensor,arr)
#change in array  ,changes in tensor?
array = array+1
#print(tensor,array)"""

#changes in tensor , in array
"""tensor = torch.ones(7)
nt = tensor.numpy()
tensor = tensor+1
print(tensor,nt)"""

#reproducbility(trying to take  random out of random)
RANDOM_SEED =42
torch.manual_seed(RANDOM_SEED)
tensor_a = torch.rand(2,3)
torch.manual_seed(RANDOM_SEED)

tensor_b  = torch.rand(2,3)
print(tensor_a)
print(tensor_b)
print(tensor_a == tensor_b)

