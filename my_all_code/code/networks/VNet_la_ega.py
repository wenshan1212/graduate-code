import torch
from torch import nn
import pdb


def gauss_kernel(channels=3, cuda=True):
    kernel_size = 3
    sigma = 1.5
    # 创建x, y, z坐标网格
    x_coord = torch.arange(kernel_size) - (kernel_size - 1) / 2
    y_coord = x_coord
    z_coord = x_coord
    x_grid, y_grid, z_grid = torch.meshgrid(x_coord, y_coord, z_coord)
    
    # 计算3D高斯核
    gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2 + z_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # 确保在正确的维度上增加批次和通道维度
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # 为每个通道重复高斯核
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)  # [C, 1, D, H, W]

    if cuda:
        gaussian_kernel = gaussian_kernel.cuda()

    return gaussian_kernel


def conv_gauss(img, kernel, stride=1):
    # Convert input image to grayscale
    img_gray = img.mean(dim=1, keepdim=True)
    # Apply convolution
    out = F.conv3d(img_gray, kernel, stride=stride, padding=1)

    return out


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, dropout_rate=0.4):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        x_out = self.dropout(x_out)  # 在注意力操作后添加 Dropout
        return x_out

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv3d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

"""
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = F.avg_pool3d(x, kernel_size=x.size()[2:])
        max_out = F.max_pool3d(x, kernel_size=x.size()[2:])
        channel_att_sum = avg_out + max_out

        # 计算通道注意力权重
        scale = torch.sigmoid(self.mlp(channel_att_sum))

        # 扩展scale张量以匹配输入张量x的形状
        scale = scale.view(scale.size(0), scale.size(1), 1, 1, 1)

        return x * scale
"""

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        # 使用最大池化和平均池化获得avg_out和max_out
        avg_out = self.mlp(F.avg_pool3d(x, kernel_size=x.size()[2:], stride=(x.size(2), x.size(3), x.size(4))))
        max_out = self.mlp(F.max_pool3d(x, kernel_size=x.size()[2:], stride=(x.size(2), x.size(3), x.size(4))))

        # 通道注意力权重
        channel_att_sum = avg_out + max_out
        scale = torch.sigmoid(channel_att_sum)

        # 使用unsqueeze和expand_as对齐维度
        scale = scale.view(scale.size(0), scale.size(1)).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)

        return x * scale


def downsample(x):
    return x[:, :, ::2, ::2, ::2]


def upsample(x, channels):
    # Step 1: Upsample in the depth dimension
    cc_d = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], device=x.device)], dim=2)
    cc_d = cc_d.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3], x.shape[4])
    
    # Step 2: Upsample in the height dimension
    cc_h = torch.cat([cc_d, torch.zeros(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3], x.shape[4], device=x.device)], dim=3)
    cc_h = cc_h.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4])
    
    # Step 3: Upsample in the width dimension
    cc_w = torch.cat([cc_h, torch.zeros(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4], device=x.device)], dim=4)
    cc_w = cc_w.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2)
    return conv_gauss(cc_w, 4 * gauss_kernel(channels))

import torch.nn.functional as F
def make_laplace(img, channels):
    img_gray = img.mean(dim=1, keepdim=True)

    # 应用3D高斯滤波
    filtered = conv_gauss(img_gray, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3] or up.shape[4] != img.shape[4]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3], img.shape[4]), mode='nearest')
    diff = img - up
    return diff
"""
class EGA(nn.Module):
    def __init__(self, in_channels, actual_in_channels=None):
        super(EGA, self).__init__()
        self.in_channels = in_channels
        self.actual_in_channels = actual_in_channels if actual_in_channels is not None else in_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.in_channels*3, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(in_channels)
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.adjust_128_to_64 = nn.Conv3d(128, 64, kernel_size=1)
        self.adjust_128_to_32 = nn.Conv3d(128, 32, kernel_size=1)
        self.adjust_128_to_16 = nn.Conv3d(128, 16, kernel_size=1)

    def forward(self, edge_feature, x, pred):
        C = x.size(1)
        # 调整边缘特征的通道数
        if edge_feature.size(1) == 1:
            edge_feature = edge_feature.repeat(1, C, 1, 1)  # 重复单通道以匹配C通道
        
        # 确保边缘特征的空间尺寸与x相匹配
        if edge_feature.size(2) != x.size(2) or edge_feature.size(3) != x.size(3):
            edge_feature = F.interpolate(edge_feature, size=x.size()[2:], mode='nearest')

        pred = torch.sigmoid(pred)
        # Reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        edge_pred = make_laplace(pred, self.in_channels)
        pred_feature = x * edge_pred
        edge_input = F.interpolate(edge_feature, size=x.size()[2:], mode='nearest')
        if x.size(1)==64 or x.size(1)==32 or x.size(1)==16:
            if x.size(1)==64:
                edge_input = self.adjust_128_to_64(edge_input)
            if x.size(1)==32:
                edge_input = self.adjust_128_to_32(edge_input)
            if x.size(1)==16:
                edge_input = self.adjust_128_to_16(edge_input)
        input_feature = x * edge_input
        
        #fusion_feature = background_x + pred_feature + input_feature
        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)


        fusion_feature = self.fusion_conv(fusion_feature)
        
        
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + x
        out = self.cbam(out)
        return out
"""
class EGA(nn.Module):
    def __init__(self, in_channels, actual_in_channels=None, dropout_rate=0.4):
        super(EGA, self).__init__()
        self.in_channels = in_channels
        self.actual_in_channels = actual_in_channels if actual_in_channels is not None else in_channels
        self.dropout = nn.Dropout3d(dropout_rate)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.in_channels*3, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(in_channels)
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.adjust_128_to_64 = nn.Conv3d(128, 64, kernel_size=1)
        self.adjust_128_to_32 = nn.Conv3d(128, 32, kernel_size=1)
        self.adjust_128_to_16 = nn.Conv3d(128, 16, kernel_size=1)

    def forward(self, edge_feature, x, pred):
        C = x.size(1)
        # 调整边缘特征的通道数
        if edge_feature.size(1) == 1:
            edge_feature = edge_feature.repeat(1, C, 1, 1)  # 重复单通道以匹配C通道
        # 确保边缘特征的空间尺寸与x相匹配
        if edge_feature.size(2) != x.size(2) or edge_feature.size(3) != x.size(3):
            edge_feature = F.interpolate(edge_feature, size=x.size()[2:], mode='nearest')

        pred = torch.sigmoid(pred)
        # Reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        edge_pred = make_laplace(pred, self.in_channels)
        pred_feature = x * edge_pred
        edge_input = F.interpolate(edge_feature, size=x.size()[2:], mode='nearest')
        
        if x.size(1)==64 or x.size(1)==32 or x.size(1)==16:
            if x.size(1)==64:
                edge_input = self.adjust_128_to_64(edge_input)
            if x.size(1)==32:
                edge_input = self.adjust_128_to_32(edge_input)
            if x.size(1)==16:
                edge_input = self.adjust_128_to_16(edge_input)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        #fusion_feature = background_x + pred_feature + input_feature
        #fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)


        fusion_feature = self.fusion_conv(fusion_feature)
        fusion_feature = self.dropout(fusion_feature)  # Applying dropout here
        
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + x
        out = self.cbam(out)
        return out



import torch.nn.functional as F

class ComputeEdgeFeature3D(nn.Module):
    def __init__(self):
        super(ComputeEdgeFeature3D, self).__init__()
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).cuda()

        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).cuda()

        sobel_kernel_z = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                                        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32).cuda()

        sobel_kernel_x = sobel_kernel_x.unsqueeze(0).repeat(1, 1, 1, 1, 1).cuda()
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0).repeat(1, 1, 1, 1, 1).cuda()
        sobel_kernel_z = sobel_kernel_z.unsqueeze(0).repeat(1, 1, 1, 1, 1).cuda()
        
        self.sobel_x = nn.Parameter(sobel_kernel_x, requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(sobel_kernel_y, requires_grad=False).cuda()
        self.sobel_z = nn.Parameter(sobel_kernel_z, requires_grad=False).cuda()

        # 将输入通道数更改为1
        self.conv_adjust = nn.Conv3d(1, 128, kernel_size=1).cuda()

    def forward(self, x):
        # 检查通道数，如果不是单通道，则将其转换为灰度图
        if x.size(1) > 1:
            x_gray = 0.299 * x[:, 0, :, :, :] + 0.587 * x[:, 1, :, :, :] + 0.114 * x[:, 2, :, :, :]
            x_gray = x_gray.unsqueeze(1)  # 增加一个通道维度
        else:
            x_gray = x
        
        # 使用conv3d函数进行卷积操作
        edge_x = F.conv3d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv3d(x_gray, self.sobel_y, padding=1)
        edge_z = F.conv3d(x_gray, self.sobel_z, padding=1)

        # 计算梯度幅度
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + edge_z ** 2)
        
        # 调整通道数为128
        edge = self.conv_adjust(edge)

        return edge



class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3, padding=1, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear",align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.4, inplace=False)
        self.dropout1 = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        if self.has_dropout:
            x4 = self.dropout(x4)
        
        x4_dw = self.block_four_dw(x4)
        
        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout1(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.dropout8 = nn.Dropout3d(p=0.2, inplace=False)
        # self.dropout7 = nn.Dropout3d(p=0.15, inplace=False)
        #self.dropout5 = nn.Dropout3d(p=0.2, inplace=False)
        self.ega = EGA(in_channels=n_filters * 8, actual_in_channels=n_filters * 8)  # EGA模块
        self.ega_1 = EGA(in_channels=n_filters * 4, actual_in_channels=n_filters * 4)  # EGA模块
        self.ega_2 = EGA(in_channels=n_filters * 2, actual_in_channels=n_filters * 2)  # EGA模块
        self.ega_3 = EGA(in_channels=n_filters , actual_in_channels=n_filters)  # EGA模块

    def forward(self, features, edge_features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        # if self.has_dropout:
        #     x5 = self.dropout5(x5)
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4
        x5_up = self.ega(edge_features, x5_up, x5_up)  # 使用EGA融合边缘特征

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        x6_up = self.ega_1(edge_features, x6_up, x6_up)  # 再次使用EGA

        x7 = self.block_seven(x6_up)
        # if self.has_dropout:
        #     x7 = self.dropout7(x7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        x7_up = self.ega_2(edge_features, x7_up, x7_up)  # 再次使用EGA

        x8 = self.block_eight(x7_up)
        # if self.has_dropout:
        #     x8 = self.dropout8(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x8_up = self.ega_3(edge_features, x8_up, x8_up)  # 最后一次使用EGA

        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.compute_edge_feature = ComputeEdgeFeature3D()  # 边缘特征提取模块
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        dim_in = 16
        feat_dim = 32
        self.pool = nn.MaxPool3d(3, stride=2)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
        
    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, input):
        edge_features = self.compute_edge_feature(input)
        features = self.encoder(input)
        out_seg, x8_up = self.decoder(features, edge_features)
        features = self.pool(features[4])
        return out_seg, features # 4, 16, 112, 112, 80


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #import pdb; pdb.set_trace()