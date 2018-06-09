import torch.nn as nn

class CONV1D3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D3, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(0.25))

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(64, 32)
        self.linear1 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)
        output = self.conv2(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)

        return output

class CONV1D2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D2, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(16, stride=16),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(64, 32)
        self.linear1 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)

        return output

class CONV1D5(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D5, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25))

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25))

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(64, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)

        return output

class CONV1D2_256_128_48(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D2_256_128_48, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(16, stride=16),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(128, 48)
        self.linear1 = nn.Linear(48, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)

        return output

class CONV1D2_64_32(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D2_64_32, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(16, stride=16),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)

        return output

class CONV1D2_256_128_64_32(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV1D2_256_128_64_32, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(16, stride=16),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(128, 64)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)
        output = self.linear2(output)

        return output

class CONV2D2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CONV2D2, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(input_size, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(16, stride=16),
            nn.Dropout(0.25))

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(8, stride=8),
            nn.Dropout(0.25))

        self.linear0 = nn.Linear(64, 32)
        self.linear1 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = self.conv0(x)
        output = self.conv1(output)

        output = output.view(output.size(0), output.size(1)*output.size(2))
        output = self.linear0(output)
        output = self.linear1(output)

        return output
