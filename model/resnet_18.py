import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, block=BasicBlock, layers=[2, 2, 2, 2], groups=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.groups = groups 
        
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, 64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(self.groups, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))  
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out) 
        
        out = self.avgpool(out)  
        out = torch.flatten(out, 1)  
        out = self.fc(out)
        
        return out

def resnet18(input_channels=3, num_classes=10, groups=32):
    return ResNet(
        input_channels=input_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        groups=groups
    )

def resnet34(input_channels=3, num_classes=10, groups=32):
    return ResNet(
        input_channels=input_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        groups=groups
    )

if __name__ == "__main__":
    x = torch.randn(32, 3, 32, 32)
    model = resnet18(input_channels=3, num_classes=10, groups=32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 
    
    batchnorm_count = 0
    groupnorm_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"Found BatchNorm layer: {name}")
            batchnorm_count += 1
        elif isinstance(module, nn.GroupNorm):
            print(f"Found GroupNorm layer: {name}")
            groupnorm_count += 1
    
    print(f"Total BatchNorm layers: {batchnorm_count}")
    print(f"Total GroupNorm layers: {groupnorm_count}")