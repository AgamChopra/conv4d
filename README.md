# PyTorch Conv4d
Simple helper functions to quickly implement simple 4d convolutions derived from pvjosue's convNd implementation(found at https://github.com/pvjosue/pytorch_convNd) and a 4d batch normalization that works by internally reshaping the 6d input data into 3d and applying torch.nn.BatchNorm1d.

main.py contains a simple 4d CNN with a `4d conv 1 layer, followed by a 4d conv downsample, and finally a transpose conv layer`. The attached image is an example of the loss visualization during a sample training run on randomly generated 4d data of shape (n,c,x1,x2,x3,x4).

Tip- Import the conv4d file as such:

      import conv4d as nn4
      
      nn4.BatchNorm4d(...) ...and so on...

## Example:

    import torch
    import conv4d as nn4

    x = torch.rand(2, 1, 10, 10, 10, 10).cuda()
    print(x.shape)

    c4d = nn4.Conv4d(in_channels=1, out_channels=6, kernel_size=2, stride=2).cuda()

    y = c4d(x)
    print(y.shape)

    cT4d = nn4.ConvTranspose4d(in_channels=6, out_channels=3, kernel_size=2, stride=2).cuda()

    y = cT4d(y)
    print(y.shape)

## Output

> torch.Size([2, 1, 10, 10, 10, 10])
> 
> torch.Size([2, 6, 5, 5, 5, 5])
> 
> torch.Size([2, 3, 10, 10, 10, 10])
