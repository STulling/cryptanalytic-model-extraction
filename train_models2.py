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

sizes = list(map(int,sys.argv[1].split("-")))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
activation_argument = sys.argv[3].lower()
activation_function = activation_function_dict[activation_argument] if len(sys.argv) > 3 else nn.ReLU

model = SimpleClassifier(sizes, activation_function)

SAMPLES = 20
np.random.seed(seed)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)
batch_size_train = 4

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
trainset = torchvision.datasets.MNIST(root='./files', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train,shuffle=True, num_workers=0)

#X = torch.tensor(np.array(np.random.normal(size=(SAMPLES, sizes[0])),dtype=np.float32))
#Y = torch.tensor(np.array(np.random.normal(size=SAMPLES)>0,dtype=np.float32))

for epoch in range(10):	# loop over the dataset multiple times
	running_loss = 0.0
	for i, (x,y) in enumerate(trainloader, 0):
	#for (x,y) in zip(X, Y):
		# get the inputs; data is a list of [inputs, labels]
		inputs = x
		#print(x.shape)
		#print(inputs[1:10])
		labels = y*0.1
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print('Finished Training')
	
# Save our amazing model.
PATH = "./models/"
file = str(seed) + "_" + "-".join(map(str,sizes)) + "_" + str(activation_argument) + ".pth"
torch.save(model, os.path.join(PATH,file))
