import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            embed_dim=256, 
            hidden_dim=256, 
            num_layers=2, 
            dropout=0.2,
            bidirectional=False,
    ):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout,
            batch_first=True, 
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, vocab_size)

    def init_hidden(self, batch_size, device=None):
        num_directions = 2 if self.bidirectional else 1
        h = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def forward(self, x, h_0=None):
        x = self.embedding(x)
        if h_0 is None:
            h_0 = self.init_hidden(x.shape[0], device=x.device)
        x, h_N = self.lstm(x, h_0)
        x = self.fc(x)
        return x, h_N
