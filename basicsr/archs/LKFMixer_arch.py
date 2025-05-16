from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
from basicsr.archs.arch_util import default_init_weights


class PLKB(nn.Module):
    '''
    Partial Large Kernel Block (PLKB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(PLKB, self).__init__()
        self.channels = channels
        self.split_channels = int(channels * split_factor)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)
        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out


class FFB(nn.Module):
    '''
    Feature Fusion Block (FFB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(FFB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.PLKB(x)
        out = self.act(self.conv1(x1 + x2))
        return out


class FDB(nn.Module):
    '''
    Feature Distillation Block (FDB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(FDB, self).__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = FFB(channels, large_kernel, split_factor)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = FFB(channels, large_kernel, split_factor)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = FFB(channels, large_kernel, split_factor)
        self.c4 = nn.Conv2d(channels, channels // 2, 1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        return out


class SFMB(nn.Module):
    '''
    Spatial Feature Modulation Block (SFMB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(SFMB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x)

        x2_1 = self.sigmoid(self.AdaptiveAvgPool(x))
        x2_2 = F.adaptive_max_pool2d(x, (x.size(2) // 8, x.size(3) // 8))
        x2_2 = self.act(self.conv1_1(self.DWConv_3(x2_2)))
        x2_2 = F.interpolate(x2_2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x2 = x2_1 * x2_2

        out = self.act(self.conv1_2(x1 + x2))
        return out


class FSB(nn.Module):
    '''
    Feature Selective Block (FSB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(FSB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_factor)
        self.DWConv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1_1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x)
        x2 = self.DWConv_3(x)
        x_fused = self.act(self.conv1_1(torch.cat((x1, x2), dim=1)))
        weight = self.sigmoid(x_fused)
        out = x1 * weight + x2 * (1 - weight)
        return out



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PixelNorm(nn.Module):
    def __init__(self, channels):
        super(PixelNorm, self).__init__()
        self.pixel_norm = nn.LayerNorm(channels)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x


class FMB(nn.Module):
    '''
    Feature Modulation Block (FMB)
    '''

    def __init__(self, channels, large_kernel, split_factor):
        super(FMB, self).__init__()
        self.FDB = FDB(channels, large_kernel, split_factor)
        self.SFMB = SFMB(channels, large_kernel, split_factor)
        self.FSB = FSB(channels, large_kernel, split_factor)

    def forward(self, input):
        out = self.FDB(input)
        out = self.SFMB(out)
        out = self.FSB(out)
        out = out + input
        return out


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Layers(nn.Module):
    def __init__(self, channels, num_block, large_kernel, split_factor):
        super(Layers, self).__init__()
        self.layers = make_layer(basic_block=FMB, num_basic_block=num_block, channels=channels,
                                 large_kernel=large_kernel, split_factor=split_factor)

    def forward(self, x):
        out = self.layers(x)
        return out


@ARCH_REGISTRY.register()
class LKFMixer(nn.Module):
    def __init__(self, in_channels, channels, out_channels, upscale, num_block, large_kernel, split_factor):
        super(LKFMixer, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layers = Layers(channels, num_block, large_kernel=large_kernel, split_factor=split_factor)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=channels, num_out_ch=out_channels)
        self.act = nn.GELU()

    def forward(self, input):
        out_fea = self.conv_first(input)
        out = self.layers(out_fea)
        out = self.act(self.conv(out))
        output = self.upsampler(out + out_fea)
        return output


# from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
# net = LKFMixer(in_channels=3, channels=48, out_channels=3, upscale=4, num_block=8, large_kernel=31, split_factor=0.25)  # 定义好的网络模型,实例化
# input = torch.randn(1, 3, 320, 180)  # 1280*720---(640, 360)---(427, 240)---(320, 180)
# print(flop_count_table(FlopCountAnalysis(net, input)))
