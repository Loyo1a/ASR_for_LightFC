import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Dual Pool Module
"""


class DPM(nn.Module):
    def __init__(self, kernel_size=3):
        super(DPM, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)

        cat_out = torch.cat([max_out, avg_out], dim=1)

        spatial_att = self.sigmoid(self.conv(cat_out))

        return x * spatial_att + x

from thop import profile
if __name__ == "__main__":
    model = DPM(kernel_size=3)
    input = torch.randn(1, 256, 20, 20)

    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

    flops, params = profile(model, inputs=(input,))
    flops=flops/1000000000
    params=params/1000000

    print(f'FLOPs: {flops} G')
    print(f'Params: {params} M')