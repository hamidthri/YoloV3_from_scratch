import torch
import torch.nn as nn

# https://medium.com/@Shahidul1004/yolov3-object-detection-f3090a24efcd

# layer     filters    size              input                output
#     0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32  0.639 BFLOPs
#     1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64  3.407 BFLOPs
#     2 conv     32  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  32  0.379 BFLOPs
#     3 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64  3.407 BFLOPs
#     4 res    1                 304 x 304 x  64   ->   304 x 304 x  64
#     5 conv    128  3 x 3 / 2   304 x 304 x  64   ->   152 x 152 x 128  3.407 BFLOPs
#     6 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
#     7 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
#     8 res    5                 152 x 152 x 128   ->   152 x 152 x 128
#     9 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
#    10 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
#    11 res    8                 152 x 152 x 128   ->   152 x 152 x 128
#    12 conv    256  3 x 3 / 2   152 x 152 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    13 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    14 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    15 res   12                  76 x  76 x 256   ->    76 x  76 x 256
#    16 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    17 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    18 res   15                  76 x  76 x 256   ->    76 x  76 x 256
#    19 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    20 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    21 res   18                  76 x  76 x 256   ->    76 x  76 x 256
#    22 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    23 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    24 res   21                  76 x  76 x 256   ->    76 x  76 x 256
#    25 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    26 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    27 res   24                  76 x  76 x 256   ->    76 x  76 x 256
#    28 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    29 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    30 res   27                  76 x  76 x 256   ->    76 x  76 x 256
#    31 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    32 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    33 res   30                  76 x  76 x 256   ->    76 x  76 x 256
#    34 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#    35 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#    36 res   33                  76 x  76 x 256   ->    76 x  76 x 256
#    37 conv    512  3 x 3 / 2    76 x  76 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    38 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    39 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    40 res   37                  38 x  38 x 512   ->    38 x  38 x 512
#    41 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    42 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    43 res   40                  38 x  38 x 512   ->    38 x  38 x 512
#    44 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    45 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    46 res   43                  38 x  38 x 512   ->    38 x  38 x 512
#    47 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    48 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    49 res   46                  38 x  38 x 512   ->    38 x  38 x 512
#    50 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    51 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    52 res   49                  38 x  38 x 512   ->    38 x  38 x 512
#    53 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    54 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    55 res   52                  38 x  38 x 512   ->    38 x  38 x 512
#    56 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    57 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    58 res   55                  38 x  38 x 512   ->    38 x  38 x 512
#    59 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    60 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    61 res   58                  38 x  38 x 512   ->    38 x  38 x 512
#    62 conv   1024  3 x 3 / 2    38 x  38 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    63 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    64 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    65 res   62                  19 x  19 x1024   ->    19 x  19 x1024
#    66 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    67 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    68 res   65                  19 x  19 x1024   ->    19 x  19 x1024
#    69 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    70 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    71 res   68                  19 x  19 x1024   ->    19 x  19 x1024
#    72 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    73 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    74 res   71                  19 x  19 x1024   ->    19 x  19 x1024
#    75 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    76 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    77 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    78 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    79 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
#    80 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
#    81 conv    255  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 255  0.189 BFLOPs
#    82 yolo
#    83 route  79
#    84 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256  0.095 BFLOPs
#    85 upsample            2x    19 x  19 x 256   ->    38 x  38 x 256
#    86 route  85 61
#    87 conv    256  1 x 1 / 1    38 x  38 x 768   ->    38 x  38 x 256  0.568 BFLOPs
#    88 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    89 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    90 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    91 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
#    92 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
#    93 conv    255  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 255  0.377 BFLOPs
#    94 yolo
#    95 route  91
#    96 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128  0.095 BFLOPs
#    97 upsample            2x    38 x  38 x 128   ->    76 x  76 x 128
#    98 route  97 36
#    99 conv    128  1 x 1 / 1    76 x  76 x 384   ->    76 x  76 x 128  0.568 BFLOPs
#   100 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#   101 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#   102 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#   103 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
#   104 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
#   105 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255  0.754 BFLOPs
#   106 yolo


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.leaky_relu(out)
        out = self.bn2(self.conv2(out))
        out = self.leaky_relu(out)
        out += x
        return out


class YoloV3(nn.Module):
    def __init__(self, num_classes):
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        # layer 0
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        # layer 1
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        # layer 2, 3
        self.res1 = ResidualBlock(64, 32)
        # layer 4 in: layer 1
        # layer 5
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        # layer 6, 7
        self.res2 = ResidualBlock(128, 64)
        # layer 8 in: layer 5
        # layer 9, 10
        self.res3 = ResidualBlock(128, 64)
        # layer 11 8
        # layer 12
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        # layer 13, 14
        self.res4 = ResidualBlock(256, 128)
        # layer 15 in: layer 12
        # layer 16, 17
        self.res5 = ResidualBlock(256, 128)
        # layer 18 in: layer 15
        # layer 19, 20
        self.res6 = ResidualBlock(256, 128)
        # layer 21 in: layer 18
        # layer 22, 23
        self.res7 = ResidualBlock(256, 128)
        # layer 24 in: layer 21
        # layer 25, 26
        self.res8 = ResidualBlock(256, 128)
        # layer 27 in: layer 24
        # layer 28, 29
        self.res9 = ResidualBlock(256, 128)
        # layer 30 in: layer 27
        # layer 31, 32
        self.res10 = ResidualBlock(256, 128)
        # layer 33 in: layer 30
        # layer 34, 35
        self.res11 = ResidualBlock(256, 128)
        # layer 36 in: layer 33
        # layer 37
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)
        # layer 38, 39
        self.res12 = ResidualBlock(512, 256)
        # layer 40 in: layer 37
        # layer 41, 42
        self.res13 = ResidualBlock(512, 256)
        # layer 43 in: layer 40
        # layer 44, 45
        self.res14 = ResidualBlock(512, 256)
        # layer 46 in: layer 43
        # layer 47, 48
        self.res15 = ResidualBlock(512, 256)
        # layer 49 in: layer 46
        # layer 50, 51
        self.res16 = ResidualBlock(512, 256)
        # layer 52 in: layer 49
        # layer 53, 54
        self.res17 = ResidualBlock(512, 256)
        # layer 55 in: layer 52
        # layer 56, 57
        self.res18 = ResidualBlock(512, 256)
        # layer 58 in: layer 55
        # layer 59, 60
        self.res19 = ResidualBlock(512, 256)
        # layer 61 in: layer 58
        # layer 62
        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(1024)
        # layer 63, 64
        self.res20 = ResidualBlock(1024, 512)
        # layer 65
        # layer 66, 67
        self.res21 = ResidualBlock(1024, 512)
        # layer 68  in: layer 65
        # layer 69, 70
        self.res22 = ResidualBlock(1024, 512)
        # layer 71 in: layer 68
        # layer 72, 73
        self.res23 = ResidualBlock(1024, 512)
        # layer 74 in: layer 71
        # layer 75
        self.conv7 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.bn7 = nn.BatchNorm2d(512)
        # layer 76
        self.conv8 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(1024)
        # layer 77
        self.conv9 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.bn9 = nn.BatchNorm2d(512)
        # layer 78
        self.conv10 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(1024)
        # layer 79
        self.conv11 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.bn11 = nn.BatchNorm2d(512)
        # --------------------------------yolo1 1--------------------------------
        # layer 80
        self.conv12 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(1024)
        # layer 81
        self.conv13 = nn.Conv2d(1024, 255, 1, 1, 0)
        self.bn13 = nn.BatchNorm2d(255)
        # layer 82 yolo
        # layer 83 route 79
        # --------------------------------yolo 1 done--------------------------------
        # --------------------------------yolo 2--------------------------------
        # layer 84
        self.conv14 = nn.Conv2d(512, 256, 1, 1, 0)
        self.bn14 = nn.BatchNorm2d(256)
        # layer 85
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        # layer 86 route 85 61
        # layer 87
        self.conv15 = nn.Conv2d(768, 256, 1, 1, 0)
        self.bn15 = nn.BatchNorm2d(256)
        # layer 88
        self.conv16 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(512)
        # layer 89
        self.conv17 = nn.Conv2d(512, 256, 1, 1, 0)
        self.bn17 = nn.BatchNorm2d(256)
        # layer 90
        self.conv18 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn18 = nn.BatchNorm2d(512)
        # layer 91
        self.conv19 = nn.Conv2d(512, 256, 1, 1, 0)
        self.bn19 = nn.BatchNorm2d(256)
        # layer 92
        self.conv20 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn20 = nn.BatchNorm2d(512)
        # layer 93
        self.conv21 = nn.Conv2d(512, 255, 1, 1, 0)
        self.bn21 = nn.BatchNorm2d(255)
        # layer 94 yolo
        # layer 95 route 91
        # --------------------------------yolo 2 done--------------------------------
        # layer 96
        self.conv22 = nn.Conv2d(255, 128, 1, 1, 0)
        self.bn22 = nn.BatchNorm2d(128)
        # layer 97
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # layer 98 route 97 36
        # layer 99
        self.conv23 = nn.Conv2d(384, 128, 1, 1, 0)
        self.bn23 = nn.BatchNorm2d(128)
        # layer 100
        self.conv24 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn24 = nn.BatchNorm2d(256)
        # layer 101
        self.conv25 = nn.Conv2d(256, 128, 1, 1, 0)
        self.bn25 = nn.BatchNorm2d(128)
        # layer 102
        self.conv26 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn26 = nn.BatchNorm2d(256)
        # layer 103
        self.conv27 = nn.Conv2d(256, 128, 1, 1, 0)
        self.bn27 = nn.BatchNorm2d(128)
        # layer 104
        self.conv28 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn28 = nn.BatchNorm2d(256)
        # layer 105
        self.conv29 = nn.Conv2d(256, 255, 1, 1, 0)
        self.bn29 = nn.BatchNorm2d(255)
        # layer 106 yolo
        # --------------------------------yolo 3--------------------------------

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.res1(x)
        x = self.bn3(self.conv3(x))
        x = self.res2(x)
        x = self.res3(x)
        x = self.bn4(self.conv4(x))
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        out1 = x
        x = self.bn5(self.conv5(x))
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)
        x = self.res17(x)
        x = self.res18(x)
        x = self.res19(x)
        
        out2 = x
        x = self.bn6(self.conv6(x))
        x = self.res20(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.res23(x)
        
        x = self.bn7(self.conv7(x))
        x = self.bn8(self.conv8(x))
        x = self.bn9(self.conv9(x))
        x = self.bn10(self.conv10(x))
        x = self.bn11(self.conv11(x))
        out3 = x
        x = self.bn12(self.conv12(x))
        x = self.bn13(self.conv13(x))
        y1 = x
        # -----y1 done-----
        x = self.bn14(self.conv14(out3))
        x = self.upsample1(x)
        x = torch.cat((x, out2), 1)
        x = self.bn15(self.conv15(x))
        x = self.bn16(self.conv16(x))
        x = self.bn17(self.conv17(x))
        x = self.bn18(self.conv18(x))
        x = self.bn19(self.conv19(x))
        out4 = x
        x = self.bn20(self.conv20(x))
        x = self.bn21(self.conv21(x))
        y2 = x
        # -----y2 done-----
        x = self.bn22(self.conv22(x))
        x = self.upsample2(x)
        x = torch.cat((x, out1), 1)
        x = self.bn23(self.conv23(x))
        x = self.bn24(self.conv24(x))
        x = self.bn25(self.conv25(x))
        x = self.bn26(self.conv26(x))
        x = self.bn27(self.conv27(x))
        x = self.bn28(self.conv28(x))
        x = self.bn29(self.conv29(x))
        y3 = x
        # -----y3 done-----
        return y1, y2, y3

# test the model

model = YoloV3(80)
x = torch.randn(1, 3, 608, 608)
y1, y2, y3 = model(x)
print(y1.shape, y2.shape, y3.shape)
