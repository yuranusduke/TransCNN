"""
Covers useful modules referred in the paper

Created by Kunhong Yu
Date: 2021/07/22
"""

import torch as t
from torch.nn import functional as F
import math
from einops import rearrange


#########################
#     0. Conv Block     #
#########################
class ConvBlock(t.nn.Module):
    """Define Conv block in the head
    """

    def __init__(self, in_channels = 3):
        """
        Args :
            --in_channels: default is 3
            --out_channels: default is 46
        """
        super(ConvBlock, self).__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU(inplace = True)
        )
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace = True)
        )

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        return x2


#########################
#       1. HMHSA        #
#########################
#*************
# 1.1 MHSA   #
#*************
class MHSA(t.nn.Module):
    """Define MHSA layer
    for a single small grid
    """

    def __init__(self, d_k, d_v, num_heads, in_channels = 128):
        """
        Args :
            --d_k: dimension for key and query
            --d_v: dimension for value
            --num_heads: attention heads
            --in_channels: default is 128
        """
        super(MHSA, self).__init__()

        self.query = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.key = t.nn.Sequential(
            t.nn.Linear(in_channels, d_k * num_heads)
        )

        self.value = t.nn.Sequential(
            t.nn.Linear(in_channels, d_v * num_heads)
        )

        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.scale = math.sqrt(self.d_k)
        self.softmax = t.nn.Softmax(dim = -1)
        self.LN = t.nn.LayerNorm(in_channels)

    def forward(self, x):
        """x has shape [m, num_groups, num_groups, g_size ** 2, c]"""
        b, num_groups, _, grid, c = x.size()
        g_size = int(math.sqrt(grid))

        # i. reshape
        x_norm = self.LN(x)

        # ii. Get key, query and value
        q = self.query(x_norm) # [m, num_groups, num_groups, h * w, d_k * num_heads]
        q = rearrange(q, 'b g1 g2 g (h d) -> b h g1 g2 g d', g1 = num_groups, g2 = num_groups,
                      g = grid, d = self.d_k, h = self.num_heads) # [m, num_heads, num_groups, num_groups, h * w, d_k]

        k = self.key(x_norm) # [m, num_groups, num_groups, h * w, d_k * num_heads]
        k = rearrange(k, 'b g1 g2 g (h d) -> b h g1 g2 g d', g1 = num_groups, g2 = num_groups,
                      g = grid, d = self.d_k, h = self.num_heads) # [m, num_heads, num_groups, num_groups, h * w, d_k]

        v = self.value(x_norm)  # [m, num_groups, num_groups, h * w, d_v * num_heads]
        v = rearrange(v, 'b g1 g2 g (h d) -> b h g1 g2 g d', g1 = num_groups, g2 = num_groups,
                      g = grid, d = self.d_v, h = self.num_heads) # [m, num_heads, num_groups, num_groups, h * w, d_v]

        # iii. LMHSA
        logit = t.einsum('b h o t i d, b h o t j d -> b h o t i j', q, k) / self.scale # [m, num_heads, num_groups, num_groups, h * w, h * w]
        attention = self.softmax(logit)
        attn_out = t.matmul(attention, v) # [m, num_heads, num_groups, num_groups, h * w, d_v]

        # iv. Reshape
        output = rearrange(attn_out, 'b h g1 g2 g d -> b g1 g2 g (h d)',
                           h = self.num_heads, g1 = num_groups, g2 = num_groups, g = grid, d = self.d_v)
        output = rearrange(output, 'b g1 g2 (s1 s2) c -> b (g1 s1) (g2 s2) c',
                           g1 = num_groups, g2 = num_groups, s1 = g_size, s2 = g_size, c = self.num_heads * self.d_v)

        return output # [m, h, w, c]


#*************
# 1.2 Groups #
#*************
class MakeGroups(t.nn.Module):
    """Define MakeGroups module
    """

    def __init__(self, g_size = 32):
        """
        Args :
            --g_size: in the paper, author uses 32 as g size
        """
        super(MakeGroups, self).__init__()

        self.g_size = g_size

    def forward(self, x):

        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        num_groups_h = h // self.g_size
        num_groups_w = w // self.g_size
        g_size_h = g_size_w = self.g_size
        x_groups = rearrange(x, 'b (num_groups_h g_size_h) (num_groups_w g_size_w) c -> b num_groups_h num_groups_w (g_size_h g_size_w) c',
                             num_groups_h = num_groups_h, num_groups_w = num_groups_w, g_size_h = g_size_h, g_size_w = g_size_w)

        return x_groups


#*************
# 1.3 HMHSA  #
#*************
class HMHSA(t.nn.Module):
    """Define HMHSA module"""

    def __init__(self, in_channels, out_channels, g_size = [8, 4, 2]):
        """
        Args :
            --in_channels
            --out_channels
            --g_size: in the paper, author uses [8, 4, 2]
        """
        super(HMHSA, self).__init__()

        assert isinstance(g_size, list) and len(g_size) == 3

        # 1. Reshape
        self.mg0 = MakeGroups(g_size = g_size[0])
        self.mg1 = MakeGroups(g_size = g_size[1])
        self.mg2 = MakeGroups(g_size = g_size[2])

        # 2. MHSA
        self.mhsa0 = MHSA(d_k = out_channels // 1, d_v = out_channels // 1, num_heads = 1, in_channels = in_channels)
        self.mhsa1 = MHSA(d_k = out_channels // 2, d_v = out_channels // 2, num_heads = 2, in_channels = in_channels)
        self.mhsa2 = MHSA(d_k = out_channels // 4, d_v = out_channels // 4, num_heads = 4, in_channels = in_channels)

        # 3. Pool
        self.max_pool0 = t.nn.MaxPool2d(kernel_size = g_size[0], stride = g_size[0])
        self.avg_pool0 = t.nn.AvgPool2d(kernel_size = g_size[0], stride = g_size[0])
        self.max_pool1 = t.nn.MaxPool2d(kernel_size = g_size[1], stride = g_size[1])
        self.avg_pool1 = t.nn.AvgPool2d(kernel_size = g_size[1], stride = g_size[1])

        # 4. Upsample
        self.upsample1 = t.nn.Upsample(scale_factor = g_size[0], mode = 'bilinear', align_corners = True)
        self.upsample2 = t.nn.Upsample(scale_factor = g_size[0] * g_size[1], mode = 'bilinear', align_corners = True)

        self.g_size = g_size

        # 5. Transformation
        self.W_p0 = t.nn.Parameter(t.randn(1, 1, 1, out_channels, in_channels), requires_grad = True)
        self.W_p1 = t.nn.Parameter(t.randn(1, 1, 1, out_channels, in_channels), requires_grad = True)
        self.W_p2 = t.nn.Parameter(t.randn(1, 1, 1, out_channels, in_channels), requires_grad = True)

        # 6. MLP
        self.mlp0 = t.nn.Sequential(
            t.nn.Linear(in_channels, in_channels * 5), # we use expansion rate 5 here
            t.nn.GELU(),
            t.nn.Linear(in_channels * 5, in_channels),
            t.nn.GELU()
        )
        self.mlp1 = t.nn.Sequential(
            t.nn.Linear(in_channels, in_channels * 5),  # we use expansion rate 5 here
            t.nn.GELU(),
            t.nn.Linear(in_channels * 5, in_channels),
            t.nn.GELU()
        )
        self.mlp2 = t.nn.Sequential(
            t.nn.Linear(in_channels, in_channels * 5),  # we use expansion rate 5 here
            t.nn.GELU(),
            t.nn.Linear(in_channels * 5, in_channels),
            t.nn.GELU()
        )

    def forward(self, x):
        """x has size [m, c, h, w]"""
        # 1. step 0
        x_0 = self.mg0(x)
        a_0 = self.mhsa0(x_0)
        a_0 = a_0.unsqueeze(dim = 3)  # [m, h, w, 1, c]
        a_0 = t.matmul(a_0, self.W_p0).squeeze().permute(0, -1, 1, 2) + x # transformation # [m, c, h, w]
        a_0_ = a_0
        a_0 = a_0.permute(0, 2, 3, 1)
        a_0 = self.mlp0(a_0)
        a_0 = a_0.permute(0, -1, 1, 2) + a_0_
        x_0 = self.max_pool0(a_0) + self.avg_pool0(a_0)

        # 2. step 1
        x_1 = self.mg1(x_0)
        a_1 = self.mhsa1(x_1)
        a_1 = a_1.unsqueeze(dim = 3)  # [m, h, w, 1, c]
        a_1 = t.matmul(a_1, self.W_p1).squeeze().permute(0, -1, 1, 2) + x_0  # transformation # [m, c, h, w]
        a_1_ = a_1
        a_1 = a_1.permute(0, 2, 3, 1)
        a_1 = self.mlp1(a_1)
        a_1 = a_1.permute(0, -1, 1, 2) + a_1_
        x_1 = self.max_pool1(a_1) + self.avg_pool1(a_1)

        # 3. step 2
        x_2 = self.mg2(x_1)
        a_2 = self.mhsa2(x_2)
        a_2 = a_2.unsqueeze(dim = 3)  # [m, h, w, 1, c]
        a_2 = t.matmul(a_2, self.W_p2).squeeze().permute(0, -1, 1, 2) + x_1  # transformation # [m, c, h, w]
        a_2_ = a_2
        a_2 = a_2.permute(0, 2, 3, 1)
        a_2 = self.mlp0(a_2)
        a_2 = a_2.permute(0, -1, 1, 2) + a_2_

        # 4. Upsample
        a_1 = self.upsample1(a_1)
        a_2 = self.upsample2(a_2)
        output = a_0 + a_1 + a_2

        return output


#########################
#        2. IRB         #
#########################
class IRB(t.nn.Module):
    """Define IRB module"""

    def __init__(self, in_channels, exp_ratio, kernel_size = 3):
        """
        Args :
            --in_channel: input channels
            --exp_ratio: expansion ratio
            --kernel_size: default is 3
        """
        super(IRB, self).__init__()

        hid_channels = int(exp_ratio * in_channels)
        self.layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = hid_channels, kernel_size = 1),
            t.nn.BatchNorm2d(hid_channels),
            t.nn.SiLU(),

            t.nn.Conv2d(in_channels = hid_channels, out_channels = hid_channels, kernel_size = kernel_size, padding = kernel_size // 2, groups = hid_channels),
            t.nn.BatchNorm2d(hid_channels),
            t.nn.SiLU(),

            t.nn.Conv2d(in_channels = hid_channels, out_channels = in_channels, kernel_size = 1),
            t.nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        x_ = self.layers(x)

        return x + x_


#########################
#        3. TDB         #
#########################
class TDB(t.nn.Module):
    """Define TDB module
    TDB to expand
    """

    def __init__(self, in_channels, out_channels):
        """
        Args :
            --in_channels: input channels
            --out_channels
        """
        super(TDB, self).__init__()

        self.branch1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1),
            t.nn.BatchNorm2d(out_channels)
        )
        self.branch2 = t.nn.Sequential(
            t.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1),
            t.nn.BatchNorm2d(out_channels)
        )

        self.silu = t.nn.SiLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = self.silu(x1 + x2)

        return x


#########################
#   4. Layer block      #
#########################
class TransCNNBlock(t.nn.Module):
    """Define TransCNN Block"""

    def __init__(self, in_channels, embed_dim, g_size, exp_ratio, kernel_size = 3):
        """
        Args :
            --in_channels: input channels
            --embed_dim
            --g_size: grid size
            --exp_ratio: expansion ratio
            --kernel_size: default is 3 for DWConv in IRB
        """
        super(TransCNNBlock, self).__init__()

        # 1. HMHSA
        self.hmhsa = HMHSA(in_channels, g_size = g_size, out_channels = embed_dim)

        # 2. IRB
        self.irb = IRB(in_channels, exp_ratio = exp_ratio, kernel_size = kernel_size)

    def forward(self, x):
        x = self.hmhsa(x)
        x = self.irb(x)

        return x
