import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from operations import *

__all__ = ['ResNet', 'resnet50', 'resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride = 1, downsample = None, search = False, op_code = 'conv3d', conv_config = None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)

		if op_code == 'conv3d':
			expand_ratio = 1
			self.convT = Conv3D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1)
		elif op_code == 'conv2d':
			expand_ratio = 1
			self.convT = Conv2D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1)
		elif op_code == 'convp3d':
			expand_ratio = 1
			self.convT = ConvP3D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1)
		elif op_code == 'convst':
			if search:
				expand_ratio = 2
			else:
				expand_ratio = 1
			self.convT = ConvST(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)

		elif op_code == 'convst_dropout':
			if search:
				expand_ratio = 1
			else:
				expand_ratio = 1
			self.convT = ConvST_Dropout(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'conv1_2d_dropout':
			if search:
				expand_ratio = 1
			else:
				expand_ratio = 1
			self.convT = Conv1_2D_Dropout(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'conv1_2_3d_dropout':
			if search:
				expand_ratio = 1
			else:
				expand_ratio = 1
			self.convT = Conv1_2_3D_Dropout(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)

		elif op_code == 'mixconv1d':
			if search:
				expand_ratio = 3
			else:
				expand_ratio = 1
			self.convT = MixConv1D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'mixconv2d':
			if search:
				expand_ratio = 3
			else:
				expand_ratio = 1
			self.convT = MixConv2D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'mixconvall':
			if search:
				expand_ratio = 7
			else:
				expand_ratio = 1
			self.convT = MixConvAll(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'conv1_2d':
			if search:
				expand_ratio = 2
			else:
				expand_ratio = 1
			self.convT = Conv1_2D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'conv2_3d':
			if search:
				expand_ratio = 2
			else:
				expand_ratio = 1
			self.convT = Conv2_3D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		elif op_code == 'conv1_2_3d':
			if search:
				expand_ratio = 3
			else:
				expand_ratio = 1
			self.convT = Conv1_2_3D(planes, planes*expand_ratio, kernel_size = 3, stride = stride, padding = 1, conv_config = conv_config)
		else:
			raise NotImplementedError

		self.conv3 = nn.Conv3d(planes*expand_ratio, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.convT(out)
		
		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(residual)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, search = False, op_code = 'conv3d', conv_config = None, conv_index = None):
		self.inplanes = 64

		self.search = search
		self.op_code = op_code
		if conv_config is None:
			conv_config = [None] * 200
		self.conv_config = conv_config
		
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
							   bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
		self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, search=search, op_code=op_code, conv_config=conv_config[:layers[0]])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, search=search, op_code=op_code, conv_config=conv_config[layers[0]:layers[0]+layers[1]])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, search=search, op_code=op_code, conv_config=conv_config[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, search=search, op_code=op_code, conv_config=conv_config[layers[0]+layers[1]+layers[2]:layers[0]+layers[1]+layers[2]+layers[3]])
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, search=False, op_code='conv3d', conv_config=None):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=(1,stride,stride), bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, search=search, op_code=op_code, conv_config=conv_config[0]))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, stride=1, downsample=None,  search=search, op_code=op_code, conv_config=conv_config[i]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.transpose(1,2).contiguous()
		x = x.view((-1,)+x.size()[2:])

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x



def resnet50(**kwargs):
	"""Constructs a ResNet-50 based model.
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	checkpoint = model_zoo.load_url(model_urls['resnet50'])

	model_dict = model.state_dict()
	new_dict = {}
	#conv_index = kwargs['conv_index']

	### ImageNet pre-trained weight:
	for k, v in checkpoint.items():
		if 'conv' in k or 'downsample.0.weight' in k:
			v = v.unsqueeze(2)
		if 'bn2' in k:
			if kwargs['op_code'] == 'conv3d' or kwargs['op_code'] == 'conv2d':
				k = k.replace('bn2', 'convT.op.1')
			elif kwargs['op_code'] == 'convp3d':
				k = k.replace('bn2', 'convT.op.2')
			elif kwargs['op_code'] == 'convst' or kwargs['op_code'] == 'conv1_2d' or kwargs['op_code'] == 'conv2_3d':
				k = k.replace('bn2', 'convT.out.0')
				if kwargs['search']:
					v = v.repeat(2)
			elif kwargs['op_code'] == 'conv1_2_3d':
				k = k.replace('bn2', 'convT.out.0')
				if kwargs['search']:
					v = v.repeat(3)

			elif kwargs['op_code'] == 'convst_dropout':
				k = k.replace('bn2', 'convT.norm')
				if kwargs['search']:
					v = v.repeat(2)
			elif kwargs['op_code'] == 'conv1_2d_dropout':
				k = k.replace('bn2', 'convT.norm')
				if kwargs['search']:
					v = v.repeat(2)
			elif kwargs['op_code'] == 'conv1_2_3d_dropout':
				k = k.replace('bn2', 'convT.norm')
				if kwargs['search']:
					v = v.repeat(3)

			elif kwargs['op_code'] == 'mixconv1d':
				k = k.replace('bn2', 'convT.out.0')
				if kwargs['search']:
					v = v.repeat(3)
			elif kwargs['op_code'] == 'mixconv2d':
				k = k.replace('bn2', 'convT.out.0')
				if kwargs['search']:
					v = v.repeat(3)
			elif kwargs['op_code'] == 'mixconvall':
				k = k.replace('bn2', 'convT.out.0')
				if kwargs['search']:
					v = v.repeat(7)
		if 'conv2' in k:
			if kwargs['op_code'] == 'conv3d' or kwargs['op_code'] == 'conv2d' or kwargs['op_code'] == 'convp3d':
				k = k.replace('conv2', 'convT.op.0')
				if kwargs['op_code'] == 'conv3d':
					v = v.repeat(1,1,3,1,1) / 3.0
				if kwargs['op_code'] == 'convp3d':
					new_dict[k.replace('convT.op.0','convT.op.1')] = model_dict[k.replace('convT.op.0','convT.op.1')]

			elif kwargs['op_code'] == 'convst' or kwargs['op_code'] == 'convst_dropout' or kwargs['op_code'] == 'conv2_3d':
				k3d = k.replace('conv2', 'convT.convs.0.0')
				k2d = k.replace('conv2', 'convT.convs.1.0')
				if kwargs['search']:
					v3d = v.repeat(1,1,3,1,1) / 3.0
					v2d = v
				else:
					cout3d, _, _, _, _ = model_dict[k3d].size()
					cout2d, _, _, _, _ = model_dict[k2d].size()
					v3d = v[:cout3d,:,:,:,:].repeat(1,1,3,1,1) / 3.0
					v2d = v[cout3d:cout3d+cout2d,:,:,:,:]
				new_dict[k3d] = v3d
				new_dict[k2d] = v2d
				continue
			elif kwargs['op_code'] == 'conv1_2d' or kwargs['op_code'] == 'conv1_2d_dropout':
				k2d = k.replace('conv2', 'convT.convs.0.0')
				k1d = k.replace('conv2', 'convT.convs.1.0')
				if kwargs['search']:
					v2d = v
					v1d = torch.sum(torch.sum(v, dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0
				else:
					cout2d, _, _, _, _ = model_dict[k2d].size()
					cout1d, _, _, _, _ = model_dict[k1d].size()
					v2d = v[:cout2d,:,:,:,:]
					v1d = torch.sum(torch.sum(v[cout2d:cout2d+cout1d,:,:,:,:], dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0
				new_dict[k2d] = v2d
				new_dict[k1d] = v1d
				continue
			elif kwargs['op_code'] == 'conv1_2_3d' or kwargs['op_code'] == 'conv1_2_3d_dropout':
				k3d = k.replace('conv2', 'convT.convs.0.0')
				k2d = k.replace('conv2', 'convT.convs.1.0')
				k1d = k.replace('conv2', 'convT.convs.2.0')
				if kwargs['search']:
					v3d = v.repeat(1,1,3,1,1) / 3.0
					v2d = v
					v1d = torch.sum(torch.sum(v, dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0
				else:
					cout3d, _, _, _, _ = model_dict[k3d].size()
					cout2d, _, _, _, _ = model_dict[k2d].size()
					try:
						cout1d, _, _, _, _ = model_dict[k1d].size()
					except:
						pass
					v3d = v[:cout3d,:,:,:,:].repeat(1,1,3,1,1) / 3.0
					v2d = v[cout3d:cout3d+cout2d,:,:,:,:]
					try:
						v1d = torch.sum(torch.sum(v[cout3d+cout2d:cout3d+cout2d+cout1d,:,:,:,:], dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0
					except:
						pass
				new_dict[k3d] = v3d
				new_dict[k2d] = v2d
				try:
					if k1d in model_dict:
						new_dict[k1d] = v1d
				except:
					pass
				continue
			elif kwargs['op_code'] == 'mixconv1d':
				### v: cout,cin,1,3,3
				k311 = k.replace('conv2', 'convT.convs.0.0')
				k131 = k.replace('conv2', 'convT.convs.1.0')
				k113 = k.replace('conv2', 'convT.convs.2.0')
				if kwargs['search']:
					#v = torch.mean(torch.mean(v, dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1)
					v = torch.sum(torch.sum(v, dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0  ### use sum to keep scale?
					v311 = v
					v131 = v.permute(0,1,3,2,4)
					v113 = v.permute(0,1,4,3,2)
				else:
					v = torch.sum(torch.sum(v, dim=3, keepdim=True), dim=4, keepdim=True).repeat(1,1,3,1,1) / 3.0
					cout311, _, _, _, _ = model_dict[k311].size()
					cout131, _, _, _, _ = model_dict[k131].size()
					cout113, _, _, _, _ = model_dict[k113].size()
					v311 = v[:cout311,:,:,:,:]
					v131 = v[cout311:cout311+cout131,:,:,:,:].permute(0,1,3,2,4)
					v113 = v[cout311+cout131:cout311+cout131+cout113,:,:,:,:].permute(0,1,3,4,2)
				new_dict[k311] = v311
				new_dict[k131] = v131
				new_dict[k113] = v113
			elif kwargs['op_code'] == 'mixconv2d':
				k331 = k.replace('conv2', 'convT.convs.0.0')
				k133 = k.replace('conv2', 'convT.convs.1.0')
				k313 = k.replace('conv2', 'convT.convs.2.0')
				if kwargs['search']:
					v331 = v.permute(0,1,4,3,2)
					v133 = v
					v313 = v.permute(0,1,3,2,4)
				else:
					cout331, _, _, _, _ = model_dict[k331].size()
					cout133, _, _, _, _ = model_dict[k133].size()
					cout313, _, _, _, _ = model_dict[k313].size()
					v331 = v[:cout331,:,:,:,:].permute(0,1,4,3,2)
					v133 = v[cout331:cout331+cout133,:,:,:,:]
					v313 = v[cout331+cout133:cout331+cout133+cout313,:,:,:,:].permute(0,1,3,2,4)
				new_dict[k331] = v331
				new_dict[k133] = v133
				new_dict[k313] = v313
				
			elif kwargs['op_code'] == 'mixconvall':
				raise NotImplementedError
		
		if 'conv3' in k:
			if kwargs['search']:
				if kwargs['op_code'] == 'convst' or kwargs['op_code'] == 'conv1_2d' or kwargs['op_code'] == 'conv2_3d':
					### 2 times
					v = v.repeat(1,2,1,1,1)
				elif kwargs['op_code'] == 'mixconv1d' or kwargs['op_code'] == 'mixconv2d' or kwargs['op_code'] == 'conv1_2_3d':
					v = v.repeat(1,3,1,1,1)
				elif kwargs['op_code'] == 'mixconvall':
					v = v.repeat(1,7,1,1,1)

					


		if k in model_dict:
			new_dict[k] = v

	model.load_state_dict(new_dict)


	return model


def resnet101(**kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		groups
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	checkpoint = model_zoo.load_url(model_urls['resnet101'])
	layer_name = list(checkpoint.keys())
	for ln in layer_name:
		if 'conv' in ln or 'downsample.0.weight' in ln:
			checkpoint[ln] = checkpoint[ln].unsqueeze(2)
		# if 'conv2' in ln:
		# 	n_out, n_in, _, _, _ = checkpoint[ln].size()
		# 	checkpoint[ln] = checkpoint[ln][:n_out // alpha * (alpha - 1), :n_in//beta,:,:,:]
	model.load_state_dict(checkpoint,strict = False)

	return model



def load_state_dict_supernet(net, super_net_dict, conv_index):
	### for convST
	print('length of conv_index:', len(conv_index))
	index = 0
	net_dict = net.state_dict()
	flag1, flag2, flag3 = False, False, False
	print(list(super_net_dict.keys()))
	for k in net_dict:
		print(index, k)
		if 'convT.convs.0.0' in k: # 3d
			if flag1 and flag2 and flag3:
				index += 1
				flag1, flag2, flag3 = False, False, False
			indices = conv_index[index]['kxkxk']
			indices.sort()
			net_dict[k] = super_net_dict[k][indices,:,:,:,:]
			flag1 = True

		if 'convT.convs.1.0' in k: # 2d
			indices = conv_index[index]['1xkxk']
			indices.sort()
			net_dict[k] = super_net_dict[k][indices,:,:,:,:]
			flag2 = True
		
		if 'convT.out.0' in k: # bn
			if 'num_batches_tracked' in k:
				continue
			channels = super_net_dict[k].shape[0] // 2
			indices0 = conv_index[index]['kxkxk']
			indices1 = [ i+channels for i in conv_index[index]['1xkxk']]
			indices = indices0 + indices1
			net_dict[k] = super_net_dict[k][indices]
			flag3 = True

	net.load_state_dict(net_dict)
	return net
