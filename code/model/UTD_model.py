import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math

class SConfig():
    def __init__(self):
        self.frame_l = 100  # the length of frames
        self.joint_n = 20  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.feat_d = 190
        self.filters = 32

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # JCD part
        self.C = SConfig()
        frame_l = self.C.frame_l
        joint_n = self.C.joint_n
        joint_d = self.C.joint_d
        feat_d = self.C.feat_d
        filters = self.C.filters
        self.dropout_ratio = 0
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1),
            spatialDropout1D(self.dropout_ratio)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(self.dropout_ratio)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(self.dropout_ratio)
        )

        # diff_slow part
        self.slow_conv1 = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, 2 * filters, 1),
            spatialDropout1D(self.dropout_ratio)
        )
        self.slow_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(self.dropout_ratio)
        )
        self.slow_conv3 = c1D(frame_l, filters, filters, 1)
        self.slow_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(self.dropout_ratio)
        )

        # fast_part
        self.fast_conv1 = nn.Sequential(
            c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(self.dropout_ratio))
        self.fast_conv2 = nn.Sequential(
            c1D(frame_l//2, 2 * filters, filters, 3), spatialDropout1D(self.dropout_ratio))
        self.fast_conv3 = nn.Sequential(
            c1D(frame_l//2, filters, filters, 1), spatialDropout1D(self.dropout_ratio))

        # after cat
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(self.dropout_ratio))

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(self.dropout_ratio))

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3), spatialDropout1D(self.dropout_ratio))

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(self.dropout_ratio)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(self.dropout_ratio)
        )

        self.linear3 = nn.Linear(128, 128)

    def forward(self, M, P=None):
        M = M.permute(0, 2, 1) # [batch, timesteps, channel]
        P = M[:,:, -self.C.joint_n*self.C.joint_d:].reshape((-1, self.C.frame_l, self.C.joint_n, self.C.joint_d))
        M = M[:, :, :-self.C.joint_n*self.C.joint_d]
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        # pool will downsample the D dim of (B,C,D)
        # but we want to downsample the C channels
        # 1x1 conv may be a better choice
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        diff_slow, diff_fast = poses_motion(P)
        x_d_slow = self.slow_conv1(diff_slow)
        x_d_slow = self.slow_conv2(x_d_slow)
        x_d_slow = self.slow_conv3(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)
        x_d_slow = self.slow_pool(x_d_slow)
        x_d_slow = x_d_slow.permute(0, 2, 1)

        x_d_fast = self.fast_conv1(diff_fast)
        x_d_fast = self.fast_conv2(x_d_fast)
        x_d_fast = self.fast_conv3(x_d_fast)
        # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

        x = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_size = 128, output_size = 40):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x



class DomainPredictor(nn.Module):
    def __init__(self, hidden_size = 512, class_num = 2, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn1_fc = nn.BatchNorm1d(int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.bn2_fc = nn.BatchNorm1d(int(hidden_size / 4))
        self.fc3 = nn.Linear(int(hidden_size / 4), class_num)
        self.prob = prob

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x



