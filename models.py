import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


CORE_TYPES = {
	"vgg19_bn": torchvision.models.vgg19_bn,
	"resnet50": torchvision.models.resnet50,
}

class FeatCore(nn.Module):
	def __init__(
		self, 
		core_type='resnet50', 
		pretrained=True, 
		finetune=True
	):
		super(FeatCore, self).__init__()
		core_loader = CORE_TYPES[core_type]
		core = core_loader(pretrained=pretrained)
		self.core = nn.Sequential(*list(core.children())[:-2])
		if not finetune:
			self.fix_weights(self.core)

	def forward(self, x):
		return self.core(x)

	def fix_weights(self, block):
		for param in block.parameters():
			param.requires_grad = False

class SimpleLinear(nn.Module):
	def __init__(self, in_shape, outdims):
		super().__init__()

		self.in_shape = in_shape
		self.outdims = outdims
		c, w, h = in_shape

		self.maxpool = nn.MaxPool2d((w,h))
		self.avgpool = nn.AvgPool2d((w,h))
		self.linear = nn.Linear(c, outdims)

	def forward(self, x):
		x = self.maxpool(x)
		x = x.view(x.shape[0], -1)
		x = self.linear(x)
		return x 
    
class Encoder(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       
    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return x 
