import torch
import torch.nn as nn
from einops import rearrange


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes= 3, places= 64)

        self.layer1 = self.make_layer(in_places= 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places= 256, places= 128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places= 512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places= 1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

def ResNet18():
    return ResNet([2, 2, 2, 2])

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

class Encoder_CNN(nn.Module):
    def __init__(self, encoder_cfg):
        super(Encoder_CNN, self).__init__()
        self.cfg = encoder_cfg
        self.d_latent = self.cfg['d_latent']
        self.res_net = self.cfg.get('res_net', 'ResNet18')
        
        # input shape b c h w, h, w = self.d_in
        if self.res_net == 'ResNet18':
            self.cnn = ResNet18()
        elif self.res_net == 'ResNet50':
            self.cnn = ResNet50()
        elif self.res_net == 'ResNet101':
            self.cnn = ResNet101()
        elif self.res_net == 'ResNet152':
            self.cnn = ResNet152()
        else:
            raise ValueError("Invalid res_net type")
        
        self.fc_out = nn.Sequential(
            nn.Linear(2048, self.d_latent),
        )
    
    def forward(self, x):
        # x embedding of the latent_shape
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x
    
if __name__ == "__main__":
    # test the model
    encoder_shape_cfg =  {
        'd_latent': 1024,
    }
   
    encoder = Encoder_CNN(encoder_shape_cfg)
    input = torch.randn(2, 3, 224, 224)
    x = encoder(input)