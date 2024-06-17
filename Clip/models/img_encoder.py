from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.res_block1 = ResidualBlock(
            in_channels=3, out_channels=16, stride=2
        )  # (16,112,112)
        self.res_block2 = ResidualBlock(
            in_channels=16, out_channels=64, stride=2
        )  # (64,56,56)
        self.res_block3 = ResidualBlock(
            in_channels=64, out_channels=256, stride=2
        )  # (256,28,28)
        self.res_block4 = ResidualBlock(
            in_channels=256, out_channels=64, stride=2
        )  # (64,14,14)
        self.res_block5 = ResidualBlock(
            in_channels=64, out_channels=16, stride=2
        )  # (16,7,7)
        self.res_block6 = ResidualBlock(
            in_channels=16, out_channels=1, stride=2
        )  # (1,4,4)

        self.w = nn.Linear(in_features=16, out_features=8)
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.w(x.veiw(x.size(0), -1))
        x = self.ln(x)

        return x
