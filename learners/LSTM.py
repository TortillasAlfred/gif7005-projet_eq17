import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(1)

class CustomLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, batch_size):
        super(CustomLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.n_fits = 0

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        self.hidden2doc = nn.Linear(hidden_dim, output_dim)

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

        self.device = torch.device("cpu")

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, sentences):
        lstm_out, self.hidden = self.lstm(sentences.reshape(40, sentences.shape[0], self.embedding_dim))
        y_pred = self.hidden2doc(lstm_out[-1].view(sentences.shape[0], -1))
        y_hat = torch.sigmoid(y_pred)

        return y_hat.view(-1)

    def partial_fit(self, sentences, y, sample_weights):
        if self.n_fits % 50 == 0:
            torch.save(self.state_dict(), "./data/LSTM/{}.pt".format(self.n_fits))
            
        self.train()
        sentences = torch.from_numpy(sentences).float().to(self.device)
        y = torch.from_numpy(y.astype(np.float16)).float()
    
        self.zero_grad()

        self.hidden = self.init_hidden()

        y_pred = self(sentences)

        loss_fn = torch.nn.BCELoss(weight=torch.from_numpy(sample_weights).float())
        loss = loss_fn(y_pred, y)
        print("BCE : {}".format(loss.item()))

        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()

        self.n_fits += 1

    def predict(self, sentences):
        self.eval()
        sentences = torch.from_numpy(sentences).float().to(self.device)

        return self(sentences)