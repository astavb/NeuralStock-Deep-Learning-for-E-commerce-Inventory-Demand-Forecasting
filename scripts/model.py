import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


class MLPModel(nn.Module):

    def __init__(self, input_size):

        super(MLPModel, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.model(x)