import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False, track_running_stats=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
        #   nn.GroupNorm(out_channels // 2, out_channels),
        nn.ReLU(inplace=True),
    ]

    #   nn.BatchNorm2d(out_channels,track_running_stats=track_running_statsResNet9,
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, args):
        super().__init__()

        self.conv1 = conv_block(
            in_channels,
            64,
        )
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes, bias=False)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def conv1x1_block(in_channels, out_channels, stride=1, track_running_stats=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
    )


class BasicBlock(nn.Module):
    """
    ResNet-18/34 BasicBlock:
      - 3x3 conv (stride can be 1 or 2)
      - 3x3 conv
      - identity skip, with 1x1 projection if shape changes
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, track_running_stats=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv1x1_block(
                in_channels, out_channels, stride=stride, track_running_stats=track_running_stats
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    A CIFAR-style ResNet-18 (3x3 stem, no initial maxpool).
    Stages: [2, 2, 2, 2] BasicBlocks with channels [64,128,256,512].
    Downsampling happens at the first block of stages 2..4 via stride=2.
    """
    def __init__(self, in_channels, num_classes, args=None, track_running_stats=False):
        super().__init__()

        # CIFAR-style stem (keeps spatial size)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        )

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  blocks=2, stride=1, track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(128, blocks=2, stride=2, track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(256, blocks=2, stride=2, track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(512, blocks=2, stride=2, track_running_stats=track_running_stats)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=False),
        )

    def _make_layer(self, out_channels, blocks, stride, track_running_stats=False):
        layers = []
        # first block may downsample (stride=2)
        layers.append(
            BasicBlock(
                self.in_planes,
                out_channels,
                stride=stride,
                track_running_stats=track_running_stats,
            )
        )
        self.in_planes = out_channels
        # remaining blocks keep size
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    out_channels,
                    stride=1,
                    track_running_stats=track_running_stats,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        return out