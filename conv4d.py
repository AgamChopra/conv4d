# Required dependency convNd.py by pvjosue at https://github.com/pvjosue/pytorch_convNd
import torch
import convNd
    
def Conv4d(in_channels: int, out_channels: int, kernel_size:int=2, 
           stride:int=1, padding:int = 0, padding_mode: str ="zeros",  
           bias: bool = True, groups: int = 1):
    w = torch.rand(1)[0]
    if bias:
        b = torch.zeros(1)[0]
    return convNd.convNd(in_channels=in_channels, out_channels=out_channels,
                           num_dims=4,kernel_size=kernel_size, 
                           stride=(stride,stride,stride,stride), padding=padding, 
                           padding_mode=padding_mode, output_padding=0,
                           is_transposed=False, use_bias=bias, groups=groups, 
                           kernel_initializer=lambda x: torch.nn.init.constant_(x, w),  
                           bias_initializer=lambda x: torch.nn.init.constant_(x, b))

def ConvTranspose4d(in_channels: int, out_channels: int, kernel_size:int=2,
                    stride:int=1, padding:int = 0, padding_mode: str ="zeros", 
                    bias: bool = True, groups: int = 1):
    w = torch.rand(1)[0]
    if bias:
        b = torch.zeros(1)[0]
    return convNd.convNd(in_channels=in_channels, out_channels=out_channels,
                           num_dims=4,kernel_size=kernel_size, 
                           stride=(stride,stride,stride,stride), padding=padding, 
                           padding_mode=padding_mode, output_padding=0,
                           is_transposed=True, use_bias=bias, groups=groups, 
                           kernel_initializer=lambda x: torch.nn.init.constant_(x, w),  
                           bias_initializer=lambda x: torch.nn.init.constant_(x, b))
