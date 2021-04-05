import sys
import os
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

PATH = sys.argv[1] #example: "./models/42_784-128-1_relu.pth"
#model = SimpleClassifier(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
testset = torchvision.datasets.MNIST(root='./files', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=True, num_workers=0)

correct = 0
total = 0
with torch.no_grad():
	for (x,y) in testloader:
		images= x 
		labels = y*0.1
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		for val in predicted:
			if labels - 0.05 <= val and val <= labels + 0.05:
				correct += 1
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))