
import numpy as np
from torch import nn
import torch.nn.functional as F
import string
import random
import torch

torch.manual_seed(97)

class FinalModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.token_embedding = nn.Embedding(vocab_size+1, embd)
    self.positional_embedding = nn.Embedding(tokens, embd)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    # Multiple Head Attention 1
    self.key1 = nn.Linear(embd, head_dim, bias=False)
    self.query1 = nn.Linear(embd, head_dim, bias=False)
    self.value1 = nn.Linear(embd, head_dim, bias=False)

     # Multiple Head Attention 2
    self.key2 = nn.Linear(embd, head_dim, bias=False)
    self.query2 = nn.Linear(embd, head_dim, bias=False)
    self.value2 = nn.Linear(embd, head_dim, bias=False)
    self.dropout = nn.Dropout(dropout)

    # Linear after Multihead
    self.ln1 = nn.Linear(embd, embd)
    torch.nn.init.uniform_(self.ln1.weight, -0.5, 0.5)
    self.nm1 = nn.LayerNorm(embd)

    #sequential prediction
    self.ln2 = nn.Linear(embd, 4*embd,)
    torch.nn.init.uniform_(self.ln2.weight, -0.5, 0.5)
    self.ac1 = nn.ReLU()
    self.ln3 = nn.Linear(4*embd, 1)
    torch.nn.init.uniform_(self.ln3.weight, -0.5, 0.5)
    self.dp1 = nn.Dropout(dropout)
    self.nm2 = nn.LayerNorm(embd)

    #prediction
    self.ln5 = nn.Linear(20, 29)
    # torch.nn.init.uniform_(self.ln5.weight, -0.05, 0.05)
    self.final = nn.Softmax(-1)



  def forward(self, x):

    token_emd = self.token_embedding(x)
    # print(token_emd, token_emd.shape, self.token_embedding.weight.shape, "token_embedding")
    posi = self.positional_embedding(torch.arange(0,tokens))
    # print(posi, posi.shape, self.positional_embedding.weight.shape,  "positional_embedding")
    x = token_emd + posi

    #Multiple Head Attention 1
    k1 = self.key1(x)
    q1 = self.query1(x)
    v1 = self.value1(x)

    wei1 = q1 @ k1.transpose(-2, -1) * head_dim**-0.5
    wei1 = wei1.masked_fill(self.tril[:tokens, :tokens] == 0, float('-inf')) # (B, T, T)
    wei1 = F.softmax(wei1, dim=-1) # (B, T, T)
    wei1 = self.dropout(wei1)
    out_1 = wei1 @ v1

    #Multiple Head Attention 2
    k2 = self.key2(x)
    q2 = self.query2(x)
    v2 = self.value2(x)

    wei2 = q2 @ k2.transpose(-2, -1) * head_dim**-0.5
    wei2 = wei2.masked_fill(self.tril[:tokens, :tokens] == 0, float('-inf')) # (B, T, T)
    wei2 = F.softmax(wei2, dim=-1) # (B, T, T)
    wei2 = self.dropout(wei2)
    out_2 = wei2 @ v2


    x_ot = torch.cat([out_1, out_2], dim=-1)

    x_ot = self.ln1(x_ot)
    x_ot = self.nm1(x_ot)

    x = x + x_ot

    #feed forward
    x_f = self.ln2(x)
    x_f = self.ac1(x_f)
    x_f = self.ln3(x_f)
    x = self.dp1(x_f)


    x = self.ln5(x.transpose(-2,-1))
    # print(x, x.shape, "*"*8)
    out = x

    # print(out, out.shape, "#"*7)



    return out
