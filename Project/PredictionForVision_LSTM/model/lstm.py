from torch import nn
from torchvision import models

class ConvLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, num_lstm_layers=1, bidirectional=True):
        super(ConvLSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        # [B, 3, 224, 224] -> [B, 16, 112, 112]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), )    #[B, 32, 56, 112]
        self.lstm1 = nn.LSTM(input_size=16 * 112 * 112,
                             hidden_size=lstm_hidden_size,
                             num_layers=num_lstm_layers,
                             batch_first=True,
                             dropout=0.5,
                             bidirectional=True)  # [B, 112, lstm_hidden_size]
        self.linear1 = nn.Sequential(nn.Linear(lstm_hidden_size * self.num_directions * num_lstm_layers, 64),
                                     nn.ReLU(inplace=True))
        self.output_layer = nn.Linear(64, 3)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return h.cuda(), c.cuda()

    def forward(self, x):
        # x shape: [B, 9, 3, 224, 224]
        B = x.size(0)
        x = x.view(B * 9, 3, 224, 224)
        output = self.conv1(x)  # [B*9, 16, 112, 112]
        output = output.view(B * 9, -1).transpose(0, 1).contiguous().view(16 * 112 * 112, B, 9)
        output = output.permute(1, 2, 0)  # -> [B, 9, 16*112*112]
        h, c = self.init_hidden(output)
        output, (h, c) = self.lstm1(output, (h, c))  # h: (num_layers * num_directions, batch, lstm_hidden_size)
        h = h.transpose_(0, 1).contiguous().view(B, -1)  # -> [B, num_layers * num_directions*lstm_hidden_size]
        output = self.linear1(h)  # [B, 64]
        output = self.output_layer(output)  # [B, 3]
        return output

class RESNET_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, num_lstm_layers=1, bidirectional=True):
        super(RESNET_LSTM, self).__init__()
        net = models.resnet18(pretrained=True)
        net.classifier = nn.Sequential()
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        # [B, 3, 224, 224] -> [B, 512, 7, 7]
        self.features = net
        self.lstm1 = nn.LSTM(input_size=512 * 7 * 7,
                             hidden_size=lstm_hidden_size,
                             num_layers=num_lstm_layers,
                             batch_first=True,
                             dropout=0.5,
                             bidirectional=bidirectional)  # [B, 7, lstm_hidden_size]
        self.linear1 = nn.Sequential(nn.Linear(lstm_hidden_size * self.num_directions * num_lstm_layers, 64),
                                     nn.ReLU(inplace=True))
        self.output_layer = nn.Linear(64, 3)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return h.cuda(), c.cuda()

    def forward(self, x):
        # x shape: [B, 9, 3, 224, 224]
        B = x.size(0)
        x = x.view(B * 9, 3, 224, 224)
        output = self.features(x)  # [B*9, 512, 7, 7]
        output = output.view(B * 9, -1).transpose(0, 1).contiguous().view(512 * 7 * 7, B, 9)
        output = output.permute(1, 2, 0)  # -> [B, 9, 512*7*7]
        h, c = self.init_hidden(output)
        output, (h, c) = self.lstm1(output, (h, c))  # h: (num_layers * num_directions, batch, lstm_hidden_size)
        h = h.transpose_(0, 1).contiguous().view(B, -1)  # -> [B, num_layers * num_directions*lstm_hidden_size]
        output = self.linear1(h)  # [B, 64]
        output = self.output_layer(output)  # [B, 3]
        return output
