# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from collections import OrderedDict
from torch import nn
import torch
import torch.optim as optim
import torchvision

class SimpleClassifier(nn.Module):
	def __init__(self, sizes, activation):
		super().__init__()
		layers = OrderedDict()
		for i in range(len(sizes) - 1):
			layers.update({f"fc{i}": nn.Linear(sizes[i],sizes[i+1])})
			layers.update({f"activation{i}": activation()})
		self.model = nn.Sequential(OrderedDict(layers))
		
	def forward(self,x):
		return self.model.forward(x)

activation_function_dict = {"relu": nn.ReLU, "elu": nn.ELU, "leaky": nn.LeakyReLU}

#Example sys.args: 784-128-1 relu 42

PATH = "./models/" 
sizes = list(map(int,sys.argv[1].split("-"))) 
activation_argument = sys.argv[2].lower() #
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42 
activation_function = activation_function_dict[activation_argument] if len(sys.argv) > 2 else nn.ReLU

model = SimpleClassifier(sizes, activation_function)

SAMPLES = 20
np.random.seed(seed)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)
batch_size_train = 4

#Transfor the mnist dataset.
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
trainset = torchvision.datasets.MNIST(root='./files', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train,shuffle=True, num_workers=0)

#Training
for epoch in range(10):	
	running_loss = 0.0
	for i, (x,y) in enumerate(trainloader, 0):
		inputs = x
		labels = y*0.1
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print('Finished Training')
	
# Save our amazing model.
file = str(seed) + "_" + "-".join(map(str,sizes)) + "_" + str(activation_argument) + ".pth"
torch.save(model.state_dict(), os.path.join(PATH,file))
