import torch
import torch.nn as nn

from time_embedding import TimeEmbedding


class ModelTest(nn.Module):
  def __init__(self, **kwargs):
    super(ModelTest, self).__init__()
    self.time_emb = TimeEmbedding(20,64)

  def forward(self, input1):
    emb = self.time_emb(input1)
    return emb


if __name__ == '__main__':
  model = ModelTest()

  x     = torch.randn([10,20]).unsqueeze(2)
  results = model(x)
  
  assert results.shape == torch.Size([10, 20, 64])