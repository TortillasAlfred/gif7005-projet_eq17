import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

class CustomLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(CustomLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2doc = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.1)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_dim),
                torch.zeros(self.num_layers, 1, self.hidden_dim))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence.view(len(sentence), 1, -1))
        y_pred = self.hidden2doc(lstm_out[-1].view(1, -1))

        return y_pred.view(-1)

    def partial_fit(self, sentences, y, sample_weights):
        self.train()
        for _ in range(3):
            self.zero_grad()

            self.hidden = self.init_hidden()

            y_pred = self(sentences)

            loss = self.loss_fn(y_pred, y)
            print("MSE : {}".format(loss.item()))

            self.optimiser.zero_grad()

            loss.backward()

            self.optimiser.step()