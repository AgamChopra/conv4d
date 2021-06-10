import torch
import conv4d
import torch.nn as nn
from matplotlib import pyplot as plt 

x = torch.rand(2, 1, 10, 10, 10, 10).cuda()
y = torch.rand(2, 3, 10, 10, 10, 10).cuda()
print(x.shape)

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.f0 = nn.Sequential(conv4d.Conv4d(in_channels=1, out_channels=3, kernel_size=1, stride=1), nn.LeakyReLU(inplace=True),conv4d.Conv4d(in_channels=3, out_channels=6, kernel_size=2, stride=2),nn.LeakyReLU(inplace=True))
        self.f1 = nn.Sequential(conv4d.ConvTranspose4d(in_channels=6, out_channels=3, kernel_size=2, stride=2),nn.Sigmoid())
        
    def forward(self, x):
        y = self.f0(x)
        y = self.f1(y)
        return y
    
model = test().cuda()
pred = model(x)
print(pred.shape)

loss_list=[]
criterion = torch.nn.SmoothL1Loss(beta = 1.0)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay= 0.0001, amsgrad=False)
for epochs in range(100):
    model.train()
    pred = model(x)
    loss = criterion(pred.squeeze(), y.squeeze())
    loss.backward()
    optimizer.step()
    print(float(loss))
    loss_list.append(float(loss))
    
plt.plot(loss_list)
plt.title('Training Loss')
plt.show()