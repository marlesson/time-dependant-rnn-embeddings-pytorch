import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    '''
    https://arxiv.org/pdf/1708.00065.pdf
    https://fridayexperiment.com/how-to-encode-time-property-in-recurrent-neutral-networks/
    '''

    def __init__(self, hidden_embedding_size, output_dim):
        super(TimeEmbedding, self).__init__()
        self.emb_weight = nn.Parameter(torch.randn(1, hidden_embedding_size)) # (1, H)
        self.emb_bias = nn.Parameter(torch.randn(hidden_embedding_size)) # (H)
        self.emb_time = nn.Parameter(torch.randn(hidden_embedding_size, output_dim)) # (H, E)

    def forward(self, input):
        # input (B, W, 1)
        x = torch.softmax(input * self.emb_weight + self.emb_bias, dim=2) # (B, W, H)
        x = torch.matmul(x, self.emb_time) # (B, W, E)
        return x