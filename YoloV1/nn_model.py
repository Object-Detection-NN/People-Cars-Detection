import torch
import torch.nn as nn

architecture_config = [
    # Tuple: (kernel_size, filters_num, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M8",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class Yolo_v1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolo_v1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs) # fully connected layers

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [ CNNBlock( in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3] ) ]
                in_channels = x[1]
            elif type(x) == str and x == "M8":
                layers += [nn.MaxPool2d(kernel_size=8, stride=8)]
            elif type(x) == str:
                layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),  # (S, S, 12) where C+B*5 = 12
        )


def _test(S=7, B=2, C=2):
    model = Yolo_v1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

if __name__ == "__main__":
    _test()


