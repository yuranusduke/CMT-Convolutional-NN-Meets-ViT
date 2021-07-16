"""
Covers useful modules referred in the paper
All dimensions in comments are induced from 224 x 224 x 3 inputs
and CMT-S

Created by Kunhong Yu
Date: 2021/07/14
"""

import torch as t
from torch.nn import functional as F
import math


#########################
#  0. Patch Aggregation #
#########################
class PatchAggregation(t.nn.Module):
    """Define Bridge/PatchAggregation module connecting each other module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 16, out_channels = 46):
        """
        Args :
            --in_channels: default is 16
            --out_channels: default is 46
        """
        super(PatchAggregation, self).__init__()

        self.pa = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                        kernel_size = 2, stride = 2),
        )
        # self.ln = t.nn.LayerNorm(out_channels)

    def forward(self, x):

        x = self.pa(x)
        b, c, h, w = x.size()
        x = F.layer_norm(x, (c, h, w))

        return x


#########################
#       1. Stem         #
#########################
class Stem(t.nn.Module):
    """Define Stem module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 3, out_channels = 16):
        """
        Args :
            --in_channels: default is 3
            --out_channels: default is 16
        """
        super(Stem, self).__init__()

        self.stem = t.nn.Sequential(
            # 1.1 One Conv layer
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 2, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU(), # 112 x 112 x 16

            # 1.2 Two subsequent Conv layers
            t.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 1, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU(), # 112 x 112 x 16
            t.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3,
                        stride = 1, padding = 1),
            t.nn.BatchNorm2d(out_channels),
            t.nn.GELU() # 112 x 112 x 16
        )

    def forward(self, x):

        x = self.stem(x)

        return x


#########################
#     3. CMT block      #
#########################
#*************
#  3.1 LPU   #
#*************
class LPU(t.nn.Module):
    """Define Local Perception Unit
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 46):
        """
        Args :
            --in_channels: default is 46
        """
        super(LPU, self).__init__()

        out_channels = in_channels
        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, groups = in_channels,
                        kernel_size = 3, stride = 1, padding = 1) # 112 x 112 x 46
        )

    def forward(self, x):

        x = x + self.dwconv(x)

        return x


#*************
#  3.2 LMHSA #
#*************
class LMHSA(t.nn.Module):
    """Define Lightweight MHSA module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, input_size, kernel_size, d_k, d_v, num_heads, in_channels = 46):
        """
        Args :
            --input_size
            --kernel_size: for DWConv
            --d_k: dimension for key and query
            --d_v: dimension for value
            --num_heads: attention heads
            --in_channels: default is 46
        """
        super(LMHSA, self).__init__()

        stride = kernel_size
        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = in_channels, groups = in_channels,
                        kernel_size = kernel_size, stride = stride)
        ) # (112 / kernel_size) x (112 x kernel_size) x 46

        self.query = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.key = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.value = t.nn.Sequential(
            t.nn.Linear(in_channels, d_v * num_heads)
        )

        self.B = t.nn.Parameter(t.rand(1, num_heads, input_size ** 2, (input_size // kernel_size) ** 2), requires_grad = True)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(self.d_k)
        self.softmax = t.nn.Softmax(dim = -1)
        self.LN = t.nn.LayerNorm(in_channels)

    def forward(self, x):
        """x has shape [m, c, h, w]"""
        b, c, h, w = x.size()
        x_ = x

        # i. reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1) # [m, h * w, c]
        x_reshape = self.LN(x_reshape)

        # ii. Get key, query and value
        q = self.query(x_reshape) # [m, h * w, d_k * num_heads]
        q = q.view(b, h * w, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [m, num_heads, h * w, d_k]

        k = self.dwconv(x) # [m, c, h', w']
        c_, h_, w_ = k.size(1), k.size(-2), k.size(-1)
        k = k.view(b, c_, h_ * w_).permute(0, 2, 1) # [m, h' * w', c]
        k = self.key(k) # [m, h' * w', d_k * num_heads]
        k = k.view(b, h_ * w_, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [m, num_heads, h' * w', d_k]

        v = self.dwconv(x)  # [m, c, h', w']
        v = v.view(b, c_, h_ * w_).permute(0, 2, 1)  # [m, h' * w', c]
        v = self.value(v)  # [m, h' * w', d_v * num_heads]
        v = v.view(b, h_ * w_, self.num_heads, self.d_v).permute(0, 2, 1, 3)  # [m, num_heads, h' * w', d_v]

        # iii. LMHSA
        logit = t.matmul(q, k.transpose(-2, -1)) / self.scale # [m, num_heads, h * w, h' * w']
        logit = logit + self.B
        attention = self.softmax(logit)
        attn_out = t.matmul(attention, v) # [m, num_heads, h * w, d_v]
        attn_out = attn_out.permute(0, 2, 1, 3) # [m, h * w, num_heads, d_v]
        attn_out = attn_out.reshape(b, h, w, self.num_heads * self.d_v).permute(0, -1, 1, 2) # [m, num_heads * d_v, h, w]

        return attn_out + x_


#*************
# 3.3 IRFFN  #
#*************
class IRFFN(t.nn.Module):
    """Define IRFNN module
    can be found in Figure 2(c) and Table 1
    """

    def __init__(self, in_channels = 46, R = 3.6):
        """
        Args :
            --in_channels: default is 46
            --R: expansion ratio, default is 3.6
        """
        super(IRFFN, self).__init__()

        exp_channels = int(in_channels * R)
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = exp_channels, kernel_size = 1),
            t.nn.BatchNorm2d(exp_channels),
            t.nn.GELU()
        ) # 112 x 112 x exp_channels

        self.dwconv = t.nn.Sequential(
            t.nn.Conv2d(in_channels = exp_channels, out_channels = exp_channels, groups = exp_channels,
                        kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(exp_channels),
            t.nn.GELU()
        ) # 112 x 112 x exp_channels

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = exp_channels, out_channels = in_channels, kernel_size = 1),
            t.nn.BatchNorm2d(in_channels)
        ) # 112 x 112 x 46

    def forward(self, x):

        _, c, h, w = x.size()
        x_ = F.layer_norm(x, (c, h, w))

        x_ = self.conv2(self.dwconv(self.conv1(x_)))

        return x + x_


#*************
#3.4 CMT block#
#*************
class CMTBlock(t.nn.Module):
    """Define CMT block"""

    def __init__(self, input_size, kernel_size, d_k, d_v, num_heads, R = 3.6, in_channels = 46):
        """
        Args :
            --input_size
            --kernel_size: for DWConv
            --d_k: dimension for key and query
            --d_v: dimension for value
            --num_heads: attention heads
            --R: expansion ratio, default is 3.6
            --in_channels: default is 46
        """
        super(CMTBlock, self).__init__()

        # 1. LPU
        self.lpu = LPU(in_channels = in_channels)

        # 2. LMHSA
        self.lmhsa = LMHSA(input_size = input_size,
                           kernel_size = kernel_size, d_k = d_k, d_v = d_v, num_heads = num_heads,
                           in_channels = in_channels)

        # 3. IRFFN
        self.irffn = IRFFN(in_channels = in_channels, R = R)

    def forward(self, x):

        x = self.lpu(x)
        x = self.lmhsa(x)
        x = self.irffn(x)

        return x
