from torch import nn
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def plot(net):
	h = 0.005
	x_min, x_max = -1, 1
	y_min, y_max = -1, 1

	xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
							torch.arange(y_min, y_max, h))
	
	in_tensor = torch.cat((xx.reshape((-1,1)), yy.reshape((-1,1))), dim=1)

	result = torch.zeros((xx.shape[0], yy.shape[1], 3))
	result2 = torch.zeros((xx.shape[0], yy.shape[1], 3))
	fig, axes = plt.subplots(nrows=2, ncols=len(list(net.parameters())[0].data) + 2, figsize=(9, 3))
	for i in range(len(list(net.parameters())[0].data)):
		weight = list(net.parameters())[0].data[i]
		bias = list(net.parameters())[1].data[i]
		z = nn.functional.relu_(torch.matmul(in_tensor, weight) + bias)
		z = z.reshape(xx.shape)
		result[:, :, i] = z
		result2[:, :, i] = (z > 0)
		#axes[i].contourf(xx, yy, z, cmap=plt.cm.coolwarm)
		axes[0, i].imshow(z)
		axes[0, i].set_xlabel('x0')
		axes[0, i].set_ylabel('x1')
		axes[1, i].imshow((z > 0))
		axes[1, i].set_xlabel('x0')
		axes[1, i].set_ylabel('x1')
		
	#z = net.forward(in_tensor)
	#z = torch.argmax(z, dim=1)
	#z = z.reshape(xx.shape)
	axes[0,len(list(net.parameters())[0].data)].set_ylabel('x1')
	axes[0,len(list(net.parameters())[0].data)].set_xlabel('x0')
	axes[0,len(list(net.parameters())[0].data)].imshow(result)
	
	axes[1,len(list(net.parameters())[0].data)].set_ylabel('x1')
	axes[1,len(list(net.parameters())[0].data)].set_xlabel('x0')
	axes[1,len(list(net.parameters())[0].data)].imshow(result2)
	
	z = net.forward(in_tensor)
	z = torch.argmax(z, dim=1)
	z = z.reshape(xx.shape)
	axes[0,len(list(net.parameters())[0].data) + 1].set_ylabel('x1')
	axes[0,len(list(net.parameters())[0].data) + 1].set_xlabel('x0')
	axes[0,len(list(net.parameters())[0].data) + 1].imshow(z)
	
	plt.show()

class SimpleClassifier(nn.Module):
	def __init__(self, dims, activation):
		super().__init__()
		layers = OrderedDict()
		for i in range(len(dims) - 1):
			layers.update({f"fc{i}": nn.Linear(dims[i],dims[i+1])})
			layers.update({f"activation{i}": activation()})
		self.model = nn.Sequential(OrderedDict(layers))
		
	def forward(self,x):
		return self.model.forward(x)
		
model = SimpleClassifier([2, 3], nn.ReLU)
print(model)

plot(model)