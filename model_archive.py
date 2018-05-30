import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()

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
