import torch
import torch.nn as nn
import torch.nn.functional as F

# upsample() 函数
"""
在PyTorch中，我们可以使用 nn.Upsample 层实现双线性插值的上采样，
其中 size 参数指定了目标大小，
mode 参数指定了插值方法，这里选择了双线性插值，
而 align_corners 参数则控制是否将坐标原点放在左上角或者中心。
"""
def upsample(dst, pre_name='', idx=0):
    """
    A customized up-sampling layer using bi-linear Interpolation
    :param dst: the target tensor, need it's size for up-sample
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :return: the up-sample layer
    """
    return nn.Upsample(size=(dst.shape[2], dst.shape[3]), mode='bilinear', align_corners=True)


# create_bn_conv() 函数
"""
在PyTorch中，我们同样可以使用 nn.Conv2d 和 nn.BatchNorm2d 层实现卷积和批归一化操作。需要注意的是，这里的输入通道数应该从 input.shape[1] 获取。
同时，如果指定了激活函数为ReLU，则需要在 nn.Sequential 中再加入一个ReLU激活层。
"""
def create_bn_conv(input, filter, kernel_size, dilation_rate=1, pre_name='',
                   idx=0, padding='same', activation='relu'):
    """
    A basic convolution-batchnormalization struct
    :param input: the input tensor
    :param filter: the filter number used in Conv layer
    :param kernel_size: the filter kernel size in Conv layer
    :param dilation_rate: the dilation rate for convolution
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    conv = nn.Conv2d(input.shape[1], filter, (kernel_size, 1), padding=padding,
                     dilation=dilation_rate)
    bn = nn.BatchNorm2d(filter)
    if activation == 'relu':
        return nn.Sequential(conv, bn, nn.ReLU(inplace=True))
    else:
        return nn.Sequential(conv, bn)


# create_u_encoder() 函数
"""
在PyTorch中，我们可以使用 nn.ModuleList 类来管理模块列表。
同时，在 forward 方法中需要手动执行每一层的计算过程，并且需要手动调用插值方法进行上采样
"""
class UEncoder(nn.Module):
    def __init__(self, filter, kernel_size, pooling_size, middle_layer_filter, depth,
                 pre_name='', idx=0, padding='same', activation='relu'):
        super(UEncoder, self).__init__()
        self.depth = depth
        self.pooling_size = pooling_size
        
        l_name = f"{pre_name}_U{idx}_enc"
        self.conv_bn0 = create_bn_conv(input, filter, kernel_size,
                                       dilation_rate=1, pre_name=l_name, idx=0,
                                       padding=padding, activation=activation)
        self.conv_bn = nn.ModuleList([create_bn_conv(self.conv_bn0, middle_layer_filter, kernel_size,
                                                      dilation_rate=1, pre_name=l_name, idx=i + 1,
                                                      padding=padding, activation=activation) for i in range(depth - 1)])
        
        self.from_encoder = nn.ModuleList()
        for d in range(depth - 1):
            self.from_encoder.append(nn.MaxPool2d((pooling_size, 1)))
            self.from_encoder.append(create_bn_conv(self.conv_bn[d], filter if d == depth - 2 else middle_layer_filter,
                                                     kernel_size, dilation_rate=1, pre_name=l_name, idx=d + 1,
                                                     padding=padding, activation=activation))
        
        self.conv_bn_depth = create_bn_conv(self.conv_bn[-1], middle_layer_filter, kernel_size,
                                            dilation_rate=1, pre_name=l_name, idx=depth,
                                            padding=padding, activation=activation)
        
        l_name = f"{pre_name}_U{idx}_dec"
        self.upsample_layers = nn.ModuleList([upsample(dst, pre_name=l_name, idx=d) for d, dst in enumerate(reversed(self.from_encoder[1::2]))])
        self.conv_bn_dec = nn.ModuleList()
        for d in range(depth - 1, 0, -1):
            ch = filter if d == 1 else middle_layer_filter
            self.conv_bn_dec.append(create_bn_conv(self.conv_bn_depth, ch, kernel_size,
                                                    dilation_rate=1, pre_name=l_name, idx=d,
                                                    padding=padding, activation=activation))
            self.conv_bn_depth = create_bn_conv(nn.Sequential(self.upsample_layers[d - 1], self.conv_bn_dec[-1],
                                                               self.from_encoder.pop()), ch, kernel_size,
                                                dilation_rate=1, pre_name=l_name, idx=d,
                                                padding=padding, activation=activation)
        
    def forward(self, input):
        conv_bn = self.conv_bn0(input)
        from_encoder = []
        for d in range(self.depth - 1):
            conv_bn = self.conv_bn[d](conv_bn)
            from_encoder.append(conv_bn)
            if d != self.depth - 2:
                conv_bn = self.from_encoder[d * 2](conv_bn)
        conv_bn = self.conv_bn_depth(conv_bn)
        
        for d in range(self.depth - 1, 0, -1):
            conv_bn = self.conv_bn_dec[d - 1](conv_bn)
            conv_bn = nn.functional.interpolate(conv_bn, size=from_encoder[-1].shape[2:], mode='bilinear', align_corners=True)
            conv_bn = self.conv_bn_dec[d](torch.cat([conv_bn, from_encoder.pop()], dim=1))
        
        return nn.functional.relu(conv_bn + self.conv_bn0(input))
    

# create_mse() 函数
class Mse(nn.Module):
    def __init__(self, filter, kernel_size, dilation_rates, pre_name='',
                 idx=0, padding='same', activation='relu'):
        super(Mse, self).__init__()
        
        l_name = f"{pre_name}_mse{idx}"
        self.convs = nn.ModuleList([create_bn_conv(input, filter, kernel_size,
                                                    dilation_rate=dr, pre_name=l_name, idx=i + 1,
                                                    padding=padding, activation=activation) for i, dr in enumerate(dilation_rates)])
        
        con_conv = torch.cat(self.convs, dim=1)
        down = create_bn_conv(con_conv, filter * 2, kernel_size, dilation_rate=1,
                              pre_name=l_name, idx=len(dilation_rates) + 1,
                              padding=padding, activation=activation)
        down = create_bn_conv(down, filter, kernel_size, dilation_rate=1,
                              pre_name=l_name, idx=len(dilation_rates) + 2,
                              padding=padding, activation=activation)
        self.out = create_bn_conv(down, filter, kernel_size, dilation_rate=1,
                                  pre_name=l_name, idx=len(dilation_rates) + 3,
                                  padding=padding, activation=None)
        
    def forward(self, input):
        convs = [conv(input) for conv in self.convs]
        con_conv = torch.cat(convs, dim=1)
        down = self.out(conv)
        return F.relu(down + input)


### demo案例
"""
在 forward 方法中，我们按照SalientSleepNet模型结构的顺序调用每一层模块，并将上一层模块的输出作为输入传递给下一层模块。
最终输出经过一个 1×1 的卷积层进行压缩。
"""
class SalientSleepNet(nn.Module):
    def __init__(self):
        super(SalientSleepNet, self).__init__()
        
        # U-Net Encoder
        self.ue1 = UEncoder(32, 5, 2, 64, 3)
        self.ue2 = UEncoder(64, 5, 2, 128, 4)
        self.ue3 = UEncoder(128, 5, 2, 256, 6)
        self.ue4 = UEncoder(256, 5, 2, 512, 3)
        
        # MSE Units
        self.mse1 = Mse(256, 7, [1, 2, 4, 8])
        self.mse2 = Mse(512, 7, [1, 2, 4, 8, 16])

        # U-Net Decoder
        self.ud4 = UEncoder(512, 5, 2, 256, 3)
        self.ud3 = UEncoder(256, 5, 2, 128, 3)
        self.ud2 = UEncoder(128, 5, 2, 64, 3)
        self.ud1 = UEncoder(64, 5, 2, 32, 2)
        
        # Output Convolutional Layer
        self.out_conv = nn.Conv2d(32, 1, (1, 1))
    
    def forward(self, input):
        # U-Net Encoder
        ue1_out = self.ue1(input)
        ue2_out = self.ue2(ue1_out)
        ue3_out = self.ue3(ue2_out)
        ue4_out = self.ue4(ue3_out)
        
        # MSE Units
        mse1_out = self.mse1(ue3_out)
        mse2_out = self.mse2(ue4_out)
        
        # U-Net Decoder
        ud4_in = torch.cat([mse2_out, ue4_out], dim=1)
        ud4_out = self.ud4(ud4_in)
        ud3_in = torch.cat([mse1_out, ue3_out], dim=1)
        ud3_out = self.ud3(torch.cat([ud4_out, ud3_in], dim=1))
        ud2_out = self.ud2(torch.cat([ud3_out, ue2_out], dim=1))
        ud1_out = self.ud1(torch.cat([ud2_out, ue1_out], dim=1))
        
        # Output Convolutional Layer
        out = self.out_conv(ud1_out)
        
        return out
