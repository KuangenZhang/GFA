import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

class Generator(nn.Module):
    def __init__(self, input_size=45 * 6):
        super(Generator, self).__init__()
        if 45 * 6 == input_size:
            hidden_size = 128
        else:
            hidden_size = 64
        # print('hidden size: ', hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout2d(p=0.1)

        self.de_fc1 = nn.Linear(hidden_size, hidden_size)
        self.de_bn1 = nn.BatchNorm1d(hidden_size)
        self.de_fc2 = nn.Linear(hidden_size, hidden_size)
        self.de_bn2 = nn.BatchNorm1d(hidden_size)
        self.de_fc3 = nn.Linear(hidden_size, hidden_size)
        self.de_bn3 = nn.BatchNorm1d(hidden_size)
        self.de_fc4 = nn.Linear(hidden_size, hidden_size)
        self.de_bn4 = nn.BatchNorm1d(hidden_size)
        self.de_fc5 = nn.Linear(hidden_size, hidden_size)
        self.de_bn5 = nn.BatchNorm1d(hidden_size)
        self.de_fc6 = nn.Linear(hidden_size, hidden_size)
        self.de_bn6 = nn.BatchNorm1d(hidden_size)
        self.de_fc7 = nn.Linear(hidden_size, input_size)

    def decode(self, x):
        x = F.relu6(self.de_bn1(self.de_fc1(x)))
        # x = self.dropout(x)
        x = F.relu6(self.de_bn2(self.de_fc2(x)))
        # x = self.dropout(x)
        x = F.relu6(self.de_bn3(self.de_fc3(x)))
        # x = self.dropout(x)
        x = F.relu6(self.de_bn4(self.de_fc4(x)))
        # x = self.dropout(x)
        x = F.relu6(self.de_bn5(self.de_fc5(x)))
        # x = self.dropout(x)
        x = F.relu6(self.de_bn6(self.de_fc6(x)))
        # x = self.dropout(x)
        x = self.de_fc7(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu6(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = F.relu6(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x = self.fc7(x)
        return x

class Classifier(nn.Module):
    def __init__(self, output_size = 19, input_size = 45*6, n_c = 1):
        super(Classifier, self).__init__()
        if 45 * 6 == input_size:
            hidden_size = 128
        else:
            hidden_size = 64
        self.n_c = n_c
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


class DomainPredictor(nn.Module):
    def __init__(self, hidden_size = 64, class_num = 2, prob=0.5):
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



