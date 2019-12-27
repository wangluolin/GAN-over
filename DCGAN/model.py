import torch.nn as nn

# define Generator Network
class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # layer1 input: 100*1*1 noise, output : (ngf*8)*4*4
        self.layer1 = self._make_layer(nz, 8*ngf, 4, 1, 0)
        # layer2 output: (ngf*4)*8*8
        self.layer2 = self._make_layer(8*ngf, 4*ngf, 4, 2, 1)
        # layer3 output: (ngf*2)*16*16
        self.layer3 = self._make_layer(4*ngf, 2*ngf, 4, 2, 1)
        # layer4 output: (ngf)*32*32
        self.layer4 = self._make_layer(2*ngf, ngf, 4, 2, 1)
        # layer5 output: 3*96*96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    def _make_layer(self, input_channels, output_channels, kernel_size, stride, padding, bias=False):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# define Discriminator Network
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # layer1: input: 3*96*96 output: ndf*32*32
        self.layer1 = self._make_layer(3, ndf, 5, 3, 1)
        # layer2: output (ndf*2)*16*16
        self.layer2 = self._make_layer(ndf, ndf*2, 4, 2, 1)
        # layer3: output (ndf*4)*8*8
        self.layer3 = self._make_layer(ndf*2, ndf*4, 4, 2, 1)
        # layer4: output (ndf*8)*4*4
        self.layer4 = self._make_layer(ndf*4, ndf*8, 4, 2, 1)
        # layer5: output a probability
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

    def _make_layer(self, input_channels, output_channels, kernel_size, stride, padding, ns=0.2, bias=False):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=ns, inplace=True)
        )

if __name__ == "__main__":
    print("haha")

        
    