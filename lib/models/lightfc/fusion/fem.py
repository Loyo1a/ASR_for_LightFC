import torch
import torch.nn as nn

def pixel_wise_corr(z, x):
    '''
    z is kernel ([32, 96, 8, 8])
    x is search ([32, 96, 16, 16])

    z -> (32, 64, 96)
    x -> (32, 96, 256)
    '''
    b, c, h, w = x.size()
    z_mat = z.contiguous().view((b, c, -1)).transpose(1, 2)  # (b,64,c)
    x_mat = x.contiguous().view((b, c, -1))  # (b,c,256)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))


class SE(nn.Module):

    def __init__(self, channels=64, reduction=1):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class LFEM(nn.Module):
    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()
        self.pw_corr = pixel_wise_corr
        self.adjust = nn.Conv2d(num_kernel, 256, 1)

    def forward(self, z, x):
        corr = self.pw_corr(z, x)
        corr = self.adjust(corr)
        #corr = torch.cat((corr, x), dim=1)
        return corr

from thop import profile
if __name__ == "__main__":
    model = LFEM(num_kernel=64, adj_channel=96)

    z = torch.randn(1, 96, 8, 8)  # template/kernel
    x = torch.randn(1, 96, 16, 16)  # search region

    # 测试前向传播
    output = model(z, x)
    print(f'z: {z.size()}')
    print(f'x: {x.size()}')
    print(f'output: {output.size()}')  # [1, 192, 16, 16]

    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(z, x))
    print(f'FLOPs: {flops / 1e9:.4f} G')
    print(f'Params: {params / 1e6:.4f} M')