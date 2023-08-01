import torch.nn as nn
from torch.cuda.amp import autocast

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual if self.downsample else x
        out = self.relu(out)
        return out
    
class ResNet50(nn.Module):
    def __init__(self, num_class, resBlock=ResBlock, repeat=[3, 4, 6, 3]):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.build_layer(resBlock, 64, repeat[0], stride=1)
        self.layer2 = self.build_layer(resBlock, 128, repeat[1], stride=2)
        self.layer3 = self.build_layer(resBlock, 256, repeat[2], stride=2)
        self.layer4 = self.build_layer(resBlock, 512, repeat[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.in_channels, num_class)

    @autocast()
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.contiguous()
        x = x.reshape(B * T, C, H, W)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.reshape(B, T, -1) # (B, T, num_class)
        return out
    
    def build_layer(self, resBlock, out_channels, repeat, stride=1):
        downsample  = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels * 4),
        ) if stride != 1 or self.in_channels != out_channels * 4 else None
        
        layers = [resBlock(self.in_channels, out_channels, stride, downsample)] + [resBlock(out_channels * 4, out_channels) for i in range(repeat-1)]
        self.in_channels = out_channels * 4
        return nn.Sequential(*layers)