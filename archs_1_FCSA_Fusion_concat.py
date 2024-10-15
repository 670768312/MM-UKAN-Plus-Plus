import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init

__all__ = [
    'KANLayer', 'KANBlock', 'DWConv', 'DW_bn_relu',
    'PatchEmbed', 'ConvLayer', 'D_ConvLayer', 'UKAN'
]




from FCS_attention import DANetHead, MultiSpectralAttentionLayer


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FcsAttention(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, reduction=16):
        super(FcsAttention, self).__init__()
        c2wh = dict([(out_channels // 4, img_size // 4), (out_channels // 2, img_size // 8), (out_channels, img_size // 16)])
        # 确保 out_channels 在 c2wh 字典中是有效的键
        if out_channels not in c2wh:
            raise ValueError(f"out_channels value {out_channels} is not supported.")

        self.spatial = SpatialAttention()
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            dct_h=c2wh[out_channels],
            dct_w=c2wh[out_channels],
            reduction=reduction,
            freq_sel_method='top16'
        )

    def forward(self, x):
        x = self.frequency_channel(x)
        x = self.spatial(x) * x
        return x

class ChannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, c)
        x = self.linear(x)
        x = x.view(b, h, w, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1, 1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)
        self.encoder4 = ConvLayer(embed_dims[0], embed_dims[1])

        self.norm0 = norm_layer(embed_dims[0] // 8)
        self.norm1 = norm_layer(embed_dims[0] // 4)
        self.norm2 = norm_layer(embed_dims[0])
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm0 = norm_layer(embed_dims[0] // 8)
        self.dnorm1 = norm_layer(embed_dims[0] // 8)
        self.dnorm2 = norm_layer(embed_dims[0] // 4)
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block01 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block12 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 4,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.block23 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer
        )])

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[3], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[4], norm_layer=norm_layer
        )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[4], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[3], norm_layer=norm_layer
        )])

        self.dblock23 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 4,
            drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer
        )])

        self.dblock12 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock01 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.patch_embed0 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=3,
                                       embed_dim=embed_dims[0] // 8)
        self.patch_embed1 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[0] // 8,
                                       embed_dim=embed_dims[0] // 4)
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[0] // 4,
                                       embed_dim=embed_dims[0])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.upsample1 = nn.Conv2d(in_channels=embed_dims[2], out_channels=embed_dims[1], kernel_size=1)
        self.upsample2 = nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[0], kernel_size=1)
        self.upsample3 = nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0] // 4, kernel_size=1)
        self.upsample4 = nn.Conv2d(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8, kernel_size=1)
        self.upsample5 = nn.Conv2d(in_channels=embed_dims[0] // 8, out_channels=embed_dims[0] // 8, kernel_size=1)

        self.FCSA1 = FcsAttention(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 4, img_size=img_size)
        self.FCSA2 = FcsAttention(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 2, img_size=img_size)
        self.FCSA3 = FcsAttention(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0] * 2, img_size=img_size)
        self.FCSA4 = FcsAttention(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1] * 2, img_size=img_size)

        self.FCSA1s = FcsAttention(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 4, img_size=img_size)
        self.FCSA2s = FcsAttention(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 2, img_size=img_size)
        self.FCSA3s = FcsAttention(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0] * 2, img_size=img_size)
        self.FCSA4s = FcsAttention(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1] * 2, img_size=img_size)

        self.backdim1s = ChannelLinear(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8)
        self.backdim2s = ChannelLinear(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 4)
        self.backdim3s = ChannelLinear(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0])
        self.backdim4s = ChannelLinear(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1])

        self.backdim1 = ChannelLinear(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8)
        self.backdim2 = ChannelLinear(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 4)
        self.backdim3 = ChannelLinear(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0])
        self.backdim4 = ChannelLinear(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1])

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]

        ### Stage 1
        out, H, W = self.patch_embed0(x)
        for i, blk in enumerate(self.block01):
            out = blk(out, H, W)
        out = self.norm0(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t1 = out
        ### Stage 2
        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block12):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        ### Stage 3
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block23):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p5 = out

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))

        # out = torch.add(out, t4)
        out = torch.cat((out, t4), dim=1)
        out = self.FCSA4s(out)
        out = self.backdim4s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p4 = out

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        # out = torch.add(out, t3)
        out = torch.cat((out, t3), dim=1)
        out = self.FCSA3s(out)
        out = self.backdim3s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p3 = out

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        # out = torch.add(out, t2)
        out = torch.cat((out, t2), dim=1)
        out = self.FCSA2s(out)
        out = self.backdim2s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock23):
            out = blk(out, H, W)

        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p2 = out

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        # out = torch.add(out, t1)
        out = torch.cat((out, t1), dim=1)
        out = self.FCSA1s(out)
        out = self.backdim1s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock12):
            out = blk(out, H, W)

        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p1 = out


        out = self.upsample1(p5)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p4), dim=1)
        out = self.FCSA4(out)
        out = self.backdim4(out)

        out = self.upsample2(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p3), dim=1)
        out = self.FCSA3(out)
        out = self.backdim3(out)

        out = self.upsample3(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p2), dim=1)
        out = self.FCSA2(out)
        out = self.backdim2(out)

        out = self.upsample4(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p1), dim=1)
        out = self.FCSA1(out)
        out = self.backdim1(out)
        
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)