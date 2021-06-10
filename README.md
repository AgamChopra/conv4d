# Conv4d
Simple helper functions to quickly implement simple 4d convolutions derived from pvjosue's convNd implementation that can be found at https://github.com/pvjosue/pytorch_convNd

## Example:

`x = torch.rand(2, 1, 10, 10, 10, 10).cuda()`

`print(x.shape)`

`conv4d = Conv4d(in_channels=1, out_channels=6, kernel_size=2, stride=2).cuda()`

`y = conv4d(x)`

`print(y.shape)`

`convT4d = ConvTranspose4d(in_channels=6, out_channels=3, kernel_size=2, stride=2).cuda()`

`y = convT4d(y)`

`print(y.shape)`

## Output

> torch.Size([2, 1, 10, 10, 10, 10])
> 
> torch.Size([2, 6, 5, 5, 5, 5])
> 
> torch.Size([2, 3, 10, 10, 10, 10])
