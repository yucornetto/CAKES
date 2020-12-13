import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import OrderedDict
class PruneBN(nn.BatchNorm3d):
    pass

class Conv3D(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(Conv3D, self).__init__()
    self.op = nn.Sequential(
      nn.Conv3d(C_in, C_out, kernel_size, stride=(1,stride,stride), padding=padding, bias=False),
      nn.BatchNorm3d(C_out),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.op(x)

class Conv2D(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(Conv2D, self).__init__()
    self.op = nn.Sequential(
      nn.Conv3d(C_in, C_out, (1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0, padding, padding), bias=False),
      nn.BatchNorm3d(C_out),
      nn.ReLU(inplace=True),
    )
  def forward(self, x):
    return self.op(x)

class ConvP3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ConvP3D, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_in, kernel_size = (1, kernel_size, kernel_size), \
                stride = (1, stride, stride), padding = (0, padding, padding), bias = False),
            nn.Conv3d(C_in, C_out, kernel_size = (kernel_size, 1, 1), \
                stride = 1, padding = (padding, 0, 0), bias = False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class ConvST(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(ConvST, self).__init__()
        C_per_conv = C_out // 2
        self.C_per_conv0 = C_out - C_per_conv
        self.C_per_conv1 = C_per_conv
        self.mask = None
        
        print('conv_config:', conv_config)

        self.C_per_convs = OrderedDict()
        if conv_config is None:
            self.C_per_convs['kxkxk'] = self.C_per_conv0
            self.C_per_convs['1xkxk'] = self.C_per_conv1
            #self.C_per_convs = {'kxkxk':self.C_per_conv0, '1xkxk':self.C_per_conv1}
        else:
            self.C_per_convs['kxkxk'] = conv_config['kxkxk']
            self.C_per_convs['1xkxk'] = conv_config['1xkxk']
            #self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = (1,stride,stride), padding = padding, bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN_select(self, thre = None):
        self.C_per_convs_pruned = {}#OrderedDict()
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        mask = weight_copy.ge(thre)
        keep_index = {}#OrderedDict()
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_index[k] = []
            for i in range(start_index, start_index + self.C_per_convs[k]):
                if mask[i]:
                    keep_index[k].append(i % self.C_per_conv0)
            print('Channel After Pruning:', len(keep_index[k]))
            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = len(keep_index[k])

        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned, keep_index

    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        mask = weight_copy.ge(thre)
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = mask[start_index:start_index+self.C_per_convs[k]].sum()
            #keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num

        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned
    


class ConvST_Dropout(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(ConvST_Dropout, self).__init__()
        C_per_conv = C_out
        self.C_per_conv0 = C_per_conv
        self.C_per_conv1 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'kxkxk':self.C_per_conv0, '1xkxk':self.C_per_conv1}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = (1,stride,stride), padding = padding, bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.norm = PruneBN(C_out*2)

        self.out = nn.Sequential(
            #PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if self.training:
            assert self.C_per_conv0 == self.C_per_conv1
            to_drop = self.C_per_conv0
            weight0 = self.convs[0][0].weight
            weight1 = self.convs[1][0].weight
            drop_num0 = random.randint(0, to_drop)
            drop_num1 = to_drop - drop_num0
            keep_num0 = self.C_per_conv0 - drop_num0
            keep_num1 = self.C_per_conv1 - drop_num1

            if keep_num0 > 0:
                keep_channels0 = random.sample(range(self.C_per_conv0), k = keep_num0)
                keep_channels0.sort()
                weight0 = weight0[keep_channels0,:,:,:,:]
            else:
                keep_channels0 = []
            
            if keep_num1 > 0:
                keep_channels1 = random.sample(range(self.C_per_conv1), k = keep_num1)
                keep_channels1.sort()
                weight1 = weight1[keep_channels1,:,:,:,:]
            else:
                keep_channels1 = []

            keep_channels0 = keep_channels0
            keep_channels1 = [i+self.C_per_conv0 for i in keep_channels1]

            keep_channels_norm = keep_channels0 + keep_channels1

            norm_weight = self.norm.weight[keep_channels_norm]
            norm_bias = self.norm.bias[keep_channels_norm]
        else:
            weight0 = self.convs[0][0].weight
            weight1 = self.convs[1][0].weight
            norm_weight = self.norm.weight
            norm_bias = self.norm.bias

        xs = []

        if keep_num0 > 0:
            x0 = F.conv3d(x,weight0,bias=self.convs[0][0].bias,stride=self.convs[0][0].stride,padding=self.convs[0][0].padding,\
                dilation=self.convs[0][0].dilation,groups=self.convs[0][0].groups)
            xs.append(x0)
        if keep_num1 > 0:
            x1 = F.conv3d(x,weight1,bias=self.convs[1][0].bias,stride=self.convs[1][0].stride,padding=self.convs[1][0].padding,\
                dilation=self.convs[1][0].dilation,groups=self.convs[1][0].groups)
            xs.append(x1)

        x = torch.cat(xs, dim = 1)
        ### check the bn momentum issue https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm3d
        x = F.batch_norm(x, self.norm.running_mean[keep_channels_norm], self.norm.running_var[keep_channels_norm], norm_weight, norm_bias, \
            self.norm.training or not self.norm.track_running_stats, self.norm.momentum, self.norm.eps)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.norm
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned

class MixConv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(MixConv1D, self).__init__()
        C_per_conv = C_out // 3
        self.C_per_conv0 = C_out - C_per_conv * 2
        self.C_per_conv1 = C_per_conv
        self.C_per_conv2 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'kx1x1':self.C_per_conv0, '1xkx1':self.C_per_conv1, '1x1xk':self.C_per_conv2}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = (1,stride,stride), padding = (padding, 0, 0), bias = False),
            ))
        
        if self.C_per_convs['1xkx1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkx1'], kernel_size = (1, kernel_size, 1), \
                stride = (1,stride,stride), padding = (0, padding, 0), bias = False),
            ))

        if self.C_per_convs['1x1xk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1x1xk'], kernel_size = (1, 1, kernel_size), \
                stride = (1,stride,stride), padding = (0, 0, padding), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned

class MixConv2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(MixConv2D, self).__init__()
        C_per_conv = C_out // 3
        self.C_per_conv0 = C_out - C_per_conv * 2
        self.C_per_conv1 = C_per_conv
        self.C_per_conv2 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'kxkx1':self.C_per_conv0, '1xkxk':self.C_per_conv1, 'kx1xk':self.C_per_conv2}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkx1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkx1'], kernel_size = (kernel_size, kernel_size, 1), \
                stride = stride, padding = (padding, padding, 0), bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = stride, padding = (0, padding, padding), bias = False),
            ))

        if self.C_per_convs['kx1xk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1xk'], kernel_size = (kernel_size, 1, kernel_size), \
                stride = stride, padding = (padding, 0, padding), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned


class MixConvAll(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(MixConvAll, self).__init__()
        C_per_conv = C_out // 7
        self.C_per_conv0 = C_out - C_per_conv * 6
        self.C_per_conv1 = C_per_conv
        self.C_per_conv2 = C_per_conv
        self.C_per_conv3 = C_per_conv
        self.C_per_conv4 = C_per_conv
        self.C_per_conv5 = C_per_conv
        self.C_per_conv6 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'kxkxk':self.C_per_conv0, '1xkxk':self.C_per_conv1, 'kx1xk':self.C_per_conv2, \
            'kxkx1':self.C_per_conv3, 'kx1x1':self.C_per_conv4, '1xkx1':self.C_per_conv5, '1x1xk':self.C_per_conv6}
        else:
            self.C_per_convs = conv_config
        print(self.C_per_convs)

        self.convs = nn.ModuleList()

        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = stride, padding = padding, bias = False),
            ))

        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = stride, padding = (0, padding, padding), bias = False),
            ))
        
        if self.C_per_convs['kx1xk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1xk'], kernel_size = (kernel_size, 1, kernel_size), \
                stride = stride, padding = (padding, 0, padding), bias = False),
            ))

        if self.C_per_convs['kxkx1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkx1'], kernel_size = (kernel_size, kernel_size, 1), \
                stride = stride, padding = (padding, padding, 0), bias = False),
            ))

        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = 1, padding = (padding, 0, 0), bias = False),
            ))

        if self.C_per_convs['1xkx1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkx1'], kernel_size = (1, kernel_size, 1), \
                stride = 1, padding = (0, padding, 0), bias = False),
            ))

        if self.C_per_convs['1x1xk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1x1xk'], kernel_size = (1, 1, kernel_size), \
                stride = 1, padding = (0, 0, padding), bias = False),
            ))

        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x

    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned


class Conv1_2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(Conv1_2D, self).__init__()
        C_per_conv = C_out // 2
        self.C_per_conv0 = C_out - C_per_conv
        self.C_per_conv1 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'1xkxk':self.C_per_conv0, 'kx1x1':self.C_per_conv1}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        
        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = (1,stride,stride), padding = (padding, 0, 0), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned

class Conv2_3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(Conv2_3D, self).__init__()
        C_per_conv = C_out // 2
        self.C_per_conv0 = C_out - C_per_conv
        self.C_per_conv1 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = {'kxkxk':self.C_per_conv0, '1xkxk':self.C_per_conv1}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = (1,stride,stride), padding = padding, bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned

class Conv1_2_3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(Conv1_2_3D, self).__init__()
        C_per_conv = C_out // 3
        self.C_per_conv0 = C_out - C_per_conv - C_per_conv
        self.C_per_conv1 = C_per_conv
        self.C_per_conv2 = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = OrderedDict()
            self.C_per_convs['kxkxk'] = self.C_per_conv0
            self.C_per_convs['1xkxk'] = self.C_per_conv1
            self.C_per_convs['kx1x1'] = self.C_per_conv2
            #self.C_per_convs = {'kxkxk':self.C_per_conv0, '1xkxk':self.C_per_conv1, 'kx1x1': self.C_per_conv2}
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = (1,stride,stride), padding = padding, bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = (1,stride,stride), padding = (padding, 0, 0), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.out = nn.Sequential(
            PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        xs = []
        for conv in self.convs:
            xs.append(conv(x))
        x = torch.cat(xs, dim = 1)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.out[0]
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned


class Conv1_2D_Dropout(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(Conv1_2D_Dropout, self).__init__()
        C_per_conv = C_out
        self.C_per_conv = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = OrderedDict()
            self.C_per_convs['1xkxk'] = self.C_per_conv
            self.C_per_convs['kx1x1'] = self.C_per_conv
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        
        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = (1,stride,stride), padding = (padding, 0, 0), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.norm = PruneBN(C_out*2)

        self.out = nn.Sequential(
            #PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if self.training:
            weight0 = self.convs[0][0].weight
            weight1 = self.convs[1][0].weight

            keep_channels = random.sample(range(self.C_per_conv * 2), k = self.C_per_conv)
            keep_channels.sort()
            keep_channels0 = [i for i in keep_channels if i < self.C_per_conv]
            keep_channels1 = [i % self.C_per_conv for i in keep_channels if i >= self.C_per_conv]
            keep_num0 = len(keep_channels0)
            keep_num1 = len(keep_channels1)

            if keep_num0 > 0:
                weight0 = weight0[keep_channels0,:,:,:,:]
   
            if keep_num1 > 0:
                weight1 = weight1[keep_channels1,:,:,:,:]

            norm_weight = self.norm.weight[keep_channels]
            norm_bias = self.norm.bias[keep_channels]

        xs = []

        if keep_num0 > 0:
            x0 = F.conv3d(x,weight0,bias=self.convs[0][0].bias,stride=self.convs[0][0].stride,padding=self.convs[0][0].padding,\
                dilation=self.convs[0][0].dilation,groups=self.convs[0][0].groups)
            xs.append(x0)
        if keep_num1 > 0:
            x1 = F.conv3d(x,weight1,bias=self.convs[1][0].bias,stride=self.convs[1][0].stride,padding=self.convs[1][0].padding,\
                dilation=self.convs[1][0].dilation,groups=self.convs[1][0].groups)
            xs.append(x1)

        x = torch.cat(xs, dim = 1)
        ### check the bn momentum issue https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm3d
        x = F.batch_norm(x, self.norm.running_mean[keep_channels], self.norm.running_var[keep_channels], norm_weight, norm_bias, \
            self.norm.training or not self.norm.track_running_stats, self.norm.momentum, self.norm.eps)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.norm
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned


class Conv1_2_3D_Dropout(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_config = None):
        super(Conv1_2_3D_Dropout, self).__init__()
        C_per_conv = C_out
        self.C_per_conv = C_per_conv

        print('conv_config:', conv_config)
        if conv_config is None:
            self.C_per_convs = OrderedDict()
            self.C_per_convs['kxkxk'] = self.C_per_conv
            self.C_per_convs['1xkxk'] = self.C_per_conv
            self.C_per_convs['kx1x1'] = self.C_per_conv
        else:
            self.C_per_convs = conv_config

        self.convs = nn.ModuleList()
        if self.C_per_convs['kxkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kxkxk'], kernel_size = kernel_size, \
                stride = (1,stride,stride), padding = padding, bias = False),
            ))
        
        if self.C_per_convs['1xkxk'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['1xkxk'], kernel_size = (1, kernel_size, kernel_size), \
                stride = (1,stride,stride), padding = (0, padding, padding), bias = False),
            ))
        if self.C_per_convs['kx1x1'] > 0:
            self.convs.append(nn.Sequential(
                nn.Conv3d(C_in, self.C_per_convs['kx1x1'], kernel_size = (kernel_size, 1, 1), \
                stride = (1,stride,stride), padding = (padding, 0, 0), bias = False),
            ))
        print(self.C_per_convs)
        print(self.convs)

        self.norm = PruneBN(C_out*3)

        self.out = nn.Sequential(
            #PruneBN(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if self.training:
            weight0 = self.convs[0][0].weight
            weight1 = self.convs[1][0].weight
            weight2 = self.convs[2][0].weight

            keep_channels = random.sample(range(self.C_per_conv*3), k = self.C_per_conv)
            keep_channels.sort()
            keep_channels0 = [i for i in keep_channels if i < self.C_per_conv]
            keep_channels1 = [i % self.C_per_conv for i in keep_channels if i >= self.C_per_conv and i < self.C_per_conv*2]
            keep_channels2 = [i % self.C_per_conv for i in keep_channels if i >= self.C_per_conv*2]
            keep_num0 = len(keep_channels0)
            keep_num1 = len(keep_channels1)
            keep_num2 = len(keep_channels2)

            if keep_num0 > 0:
                weight0 = weight0[keep_channels0,:,:,:,:]
   
            if keep_num1 > 0:
                weight1 = weight1[keep_channels1,:,:,:,:]
            
            if keep_num2 > 0:
                weight2 = weight2[keep_channels2,:,:,:,:]

            norm_weight = self.norm.weight[keep_channels]
            norm_bias = self.norm.bias[keep_channels]

        xs = []

        if keep_num0 > 0:
            x0 = F.conv3d(x,weight0,bias=self.convs[0][0].bias,stride=self.convs[0][0].stride,padding=self.convs[0][0].padding,\
                dilation=self.convs[0][0].dilation,groups=self.convs[0][0].groups)
            xs.append(x0)
        if keep_num1 > 0:
            x1 = F.conv3d(x,weight1,bias=self.convs[1][0].bias,stride=self.convs[1][0].stride,padding=self.convs[1][0].padding,\
                dilation=self.convs[1][0].dilation,groups=self.convs[1][0].groups)
            xs.append(x1)
        if keep_num2 > 0:
            x2 = F.conv3d(x,weight2,bias=self.convs[2][0].bias,stride=self.convs[2][0].stride,padding=self.convs[2][0].padding,\
                dilation=self.convs[2][0].dilation,groups=self.convs[2][0].groups)
            xs.append(x2)

        x = torch.cat(xs, dim = 1)
        ### check the bn momentum issue https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm3d
        x = F.batch_norm(x, self.norm.running_mean[keep_channels], self.norm.running_var[keep_channels], norm_weight, norm_bias, \
            self.norm.training or not self.norm.track_running_stats, self.norm.momentum, self.norm.eps)
        x = self.out(x)
        return x
    
    def pruneBN(self, thre = None):
        self.C_per_convs_pruned = {}
        m = self.norm
        total = m.weight.data.shape[0]
        bn = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(total / len(self.convs))
        if thre is None:
            thre = y[-thre_index]
        weight_copy = m.weight.data.abs().clone()

        start_index = 0
        for k in self.C_per_convs:
            print('Operation: ', k)
            print('Channel Before Pruning:', self.C_per_convs[k])
            keep_num = weight_copy[start_index:start_index+self.C_per_convs[k]].ge(thre).sum()
            print('Channel After Pruning:', keep_num)

            start_index += self.C_per_convs[k]
            self.C_per_convs_pruned[k] = keep_num
        
        print('After pruning:', self.C_per_convs_pruned)
        return self.C_per_convs_pruned
