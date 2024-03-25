import torch
import torch.nn as nn
import numpy as np
import math

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
#
#
class LSKblockAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
#     def __init__(self, dim):
#         super().__init__()
#
#         self.depth = math.ceil(dim / 8) * 8
#         self.dim2 = self.depth // 4
#         self.conv11 = nn.Conv2d(dim, self.depth, kernel_size=1, padding=0)
#         # block1:
#         self.conv_0 = nn.Conv2d(self.dim2, self.dim2, 5, padding=2, groups=self.dim2)
#         self.conv0_1 = nn.Conv2d(self.dim2, self.dim2, (1, 7), padding=(0, 3), groups=self.dim2)
#         self.conv0_2 = nn.Conv2d(self.dim2, self.dim2, (7, 1), padding=(3, 0), groups=self.dim2)
#
#         self.conv1_1 = nn.Conv2d(self.dim2, self.dim2, (1, 11), padding=(0, 5), groups=self.dim2)
#         self.conv1_2 = nn.Conv2d(self.dim2, self.dim2, (11, 1), padding=(5, 0), groups=self.dim2)
#
#         self.conv2_1 = nn.Conv2d(self.dim2, self.dim2, (1, 21), padding=(0, 10), groups=self.dim2)
#         self.conv2_2 = nn.Conv2d(self.dim2, self.dim2, (21, 1), padding=(10, 0), groups=self.dim2)
#         self.conv3 = nn.Conv2d(self.dim2, dim, 1)
#
#         # block2:
#         self.conv0 = nn.Conv2d(self.dim2, self.dim2, 5, padding=2, groups=self.dim2)  # 输出大小不变，
#         self.convl = nn.Conv2d(self.dim2, self.dim2, 7, stride=1, padding=9, groups=self.dim2, dilation=3)
#         self.conv0_s = nn.Conv2d(self.dim2, self.dim2 // 2, 1)
#         self.conv1_s = nn.Conv2d(self.dim2, self.dim2 // 2, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv_m = nn.Conv2d(self.dim2 // 2, self.dim2, 1)
#
#
#     def forward(self, x):
#         u = x.clone()
#         # 判断X.dim是否为4的倍数
#         feature = self.conv11(x)  # 通道
#         # depth = x.shape[1]
#         # if depth % 4 == 0:
#         #     x1, x2, x3, x4 = torch.split(x, 4, dim=1)
#         # else:
#         #     depth = math.ceil(x.shape[1] / 4) * 4
#         #     feature = nn.Conv2d()
#         [x1, x2, x3, x4] = torch.chunk(feature, 4, dim=1)
#         # print("feature:",feature[1].shape)
#         # x1,x2,x3,x4=feature[0],feature
#         # else:
#         #     depth = math.ceil(depth / 4) * 4
#         #     feature =self.conv11(x)#通道变为4的倍数
#         #     feature1 = torch.spilt(x, 4, dim=1)
#         #     x1,x2,x3,x4=feature1
#
#         y1 = self.conv_0(x1)
#         y2 = self.conv0_1(x2)
#         y3 = self.conv0_2(x3)
#
#         attn_0 = self.conv0_1(y1)
#         attn_0 = self.conv0_2(attn_0)
#
#         attn_1 = self.conv1_1(y2)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_2 = self.conv2_1(y3)
#         attn_2 = self.conv2_2(attn_2)
#         #print("attn_2:", attn_2.shape)
#         attn_3 = y1 + y2 + y3 + attn_0 + attn_1 + attn_2
#         #print("attn_3:", attn_3.shape)
#
#         # block2:
#
#         attnt1 = self.conv0(x4)
#         attnt2 = self.convl(attnt1)
#
#         attnt1 = self.conv0_s(attnt1)
#         attnt2 = self.conv1_s(attnt2)
#
#         attnt = torch.cat([attnt1, attnt2], dim=1)
#         avg_attn = torch.mean(attnt, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attnt, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         attnt = attnt1 * sig[:, 0, :, :].unsqueeze(1) + attnt2 * sig[:, 1, :, :].unsqueeze(1)
#         attnt = self.conv_m(attnt)
#         #print("attnt:", attnt.shape)
#         attn = attnt + attn_3
#         #print("attn:", attn.shape)
#         attn = self.conv3(attn)
#         return u + (u * attn)

    #
    # def __init__(self, dim,c=2, eps=1e-5):
    #     super().__init__()
    #     self.a = torch.tensor([0.1], requires_grad=True)
    #     self.b = torch.tensor([0.1], requires_grad=True)
    #     self.c = torch.tensor([0.1], requires_grad=True)
    #     self.d = torch.tensor([0.1], requires_grad=True)
    #     self.f = torch.tensor([0.1], requires_grad=True)
    #     self.e = torch.tensor([0.1], requires_grad=True)
    #     self.depth = math.ceil(dim / 8) * 8
    #     self.dim2 = self.depth // 4
    #     self.conv11 = nn.Conv2d(dim, self.depth, kernel_size=1, padding=0)
    #     # block1:
    #     self.conv_0 = nn.Conv2d(self.dim2, self.dim2, 3, padding=1, groups=self.dim2)
    #     self.conv0_1 = nn.Conv2d(self.dim2, self.dim2, (1, 5), padding=(0, 2), groups=self.dim2)
    #     self.conv0_2 = nn.Conv2d(self.dim2, self.dim2, (5, 1), padding=(2, 0), groups=self.dim2)
    #
    #     self.conv1_1 = nn.Conv2d(self.dim2, self.dim2, (1, 7), padding=(0, 3), groups=self.dim2)
    #     self.conv1_2 = nn.Conv2d(self.dim2, self.dim2, (7, 1), padding=(3, 0), groups=self.dim2)
    #
    #     self.conv2_1 = nn.Conv2d(self.dim2, self.dim2, (1, 11), padding=(0, 5), groups=self.dim2)
    #     self.conv2_2 = nn.Conv2d(self.dim2, self.dim2, (11, 1), padding=(5, 0), groups=self.dim2)
    #     self.conv3 = nn.Conv2d(self.dim2, dim, 1)
    #
    #     # block2:
    #     self.conv0 = nn.Conv2d(self.dim2, self.dim2, 5, padding=2, groups=self.dim2)  # 输出大小不变，
    #     self.convl = nn.Conv2d(self.dim2, self.dim2, 7, stride=1, padding=9, groups=self.dim2, dilation=3)
    #     self.conv0_s = nn.Conv2d(self.dim2, self.dim2 // 2, 1)
    #     self.conv1_s = nn.Conv2d(self.dim2, self.dim2 // 2, 1)
    #     self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
    #     self.conv_m = nn.Conv2d(self.dim2 // 2, self.dim2, 1)
    #
    #     self.avgpool = nn.AdaptiveAvgPool2d(1)
    #     self.eps = eps
    #     self.c = c
    #
    #
    #
    # def forward(self, x):
    #     u = x.clone()
    #     # 判断X.dim是否为4的倍数
    #     feature = self.conv11(x)  # 通道
    #     # depth = x.shape[1]
    #     # if depth % 4 == 0:
    #     #     x1, x2, x3, x4 = torch.split(x, 4, dim=1)
    #     # else:
    #     #     depth = math.ceil(x.shape[1] / 4) * 4
    #     #     feature = nn.Conv2d()
    #     [x1, x2, x3, x4] = torch.chunk(feature, 4, dim=1)
    #     # print("feature:",feature[1].shape)
    #     # x1,x2,x3,x4=feature[0],feature
    #     # else:
    #     #     depth = math.ceil(depth / 4) * 4
    #     #     feature =self.conv11(x)#通道变为4的倍数
    #     #     feature1 = torch.spilt(x, 4, dim=1)
    #     #     x1,x2,x3,x4=feature1
    #
    #     y1 = self.conv_0(x1)
    #     y2 = self.conv0_1(x2)
    #     y3 = self.conv0_2(x3)
    #
    #     attn_0 = self.conv0_1(y1)
    #     attn_0 = self.conv0_2(attn_0)
    #     attn_0_0 = self.avgpool(attn_0)
    #     mean_1=attn_0_0.mean(dim=1, keepdim=True)
    #     mean_2 = (attn_0_0 ** 2).mean(dim=1, keepdim=True)
    #     var_1 = mean_2 - mean_1 ** 2
    #     attn_0_norm = (attn_0_0 - mean_1) / torch.sqrt(var_1 + self.eps)
    #     attn_0_transform = torch.exp(-(attn_0_norm ** 2 / 2 * self.c))
    #     attn_0=attn_0*attn_0_transform.expand_as(attn_0)
    #
    #     attn_1 = self.conv1_1(y2)
    #     attn_1 = self.conv1_2(attn_1)
    #
    #     attn_1_1 = self.avgpool(attn_1)
    #     mean1_1 = attn_1_1.mean(dim=1, keepdim=True)
    #     mean1_2 = (attn_1_1 ** 2).mean(dim=1, keepdim=True)
    #     var_2 = mean1_2 - mean1_1 ** 2
    #     attn_1_norm = (attn_1_1 - mean1_1) / torch.sqrt(var_2 + self.eps)
    #     attn_1_transform = torch.exp(-(attn_1_norm ** 2 / 2 * self.c))
    #     attn_1 = attn_1 * attn_1_transform.expand_as(attn_1)
    #
    #
    #     attn_2 = self.conv2_1(y3)
    #     attn_2 = self.conv2_2(attn_2)
    #
    #     attn_2_2 = self.avgpool(attn_2)
    #     mean2_1 = attn_2_2.mean(dim=1, keepdim=True)
    #     mean2_2 = (attn_2_2 ** 2).mean(dim=1, keepdim=True)
    #     var_3 = mean2_2 - mean1_1 ** 2
    #     attn_2_norm = (attn_2_2 - mean2_1) / torch.sqrt(var_2 + self.eps)
    #     attn_2_transform = torch.exp(-(attn_2_norm ** 2 / 2 * self.c))
    #     attn_2 = attn_2 * attn_2_transform.expand_as(attn_2)
    #
    #     #print("attn_2:", attn_2.shape)
    #     attn_3 = y1 + y2 + y3 + attn_0 + attn_1 + attn_2
    #     #print("attn_3:", attn_3.shape)
    #
    #     # block2:
    #
    #     attnt1 = self.conv0(x4)
    #     attnt2 = self.convl(attnt1)
    #
    #     attnt1 = self.conv0_s(attnt1)
    #     attnt2 = self.conv1_s(attnt2)
    #
    #     attnt = torch.cat([attnt1, attnt2], dim=1)
    #     avg_attn = torch.mean(attnt, dim=1, keepdim=True)
    #     max_attn, _ = torch.max(attnt, dim=1, keepdim=True)
    #     agg = torch.cat([avg_attn, max_attn], dim=1)
    #     sig = self.conv_squeeze(agg).sigmoid()
    #     attnt = attnt1 * sig[:, 0, :, :].unsqueeze(1) + attnt2 * sig[:, 1, :, :].unsqueeze(1)
    #     attnt = self.conv_m(attnt)
    #     #print("attnt:", attnt.shape)
    #     attn = attnt + attn_3
    #     #print("attn:", attn.shape)
    #     attn = self.conv3(attn)
    #     return u + (u * attn)
# if __name__ == '__main__':
#     x = torch.randn(1, 512, 640, 640)
#     model = ma_ls(512)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"模型的参数量：{total_params}")
#     y = model(x)
#
#     print(y.shape)
