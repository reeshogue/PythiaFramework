import torch
import math
from attn import Attention
from torch.cuda import amp



class SelfAttention(torch.nn.Module):
    def __init__(self, size=64*64*3, nheads=4):
        super().__init__()
        self.size = size
        self.nheads = nheads
        self.linear_q = torch.nn.Linear(self.size, self.size*self.nheads, bias=False)
        self.linear_k = torch.nn.Linear(self.size, self.size*self.nheads, bias=False)
        self.linear_v = torch.nn.Linear(self.size, self.size*self.nheads, bias=False)

        self.attn_fn = Attention(size)
        self.norm = torch.nn.LayerNorm((size,))
    def forward(self, x : torch.Tensor):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attended = self.attn_fn(q, k, v)
        outp = sum(torch.split(attended, self.size, -1))
        return self.norm(outp) + x

class FFN(torch.nn.Module):
    def __init__(self, size=64*64*3, outsize=None):
        super().__init__()
        self.size = size
        if outsize is None:
            outsize = self.size
        self.linear_a = torch.nn.Linear(size, size)
        self.gelu = torch.nn.GELU()
        self.linear_b = torch.nn.Linear(size, outsize)
        self.norm = torch.nn.LayerNorm((outsize,))
    def forward(self, x):
        y = self.linear_a(x)
        y = self.gelu(y)
        y = self.linear_b(y)
        y = self.norm(y) + y
        return x

class RecurrentPositionalEmbedddings(torch.nn.Module):
    def __init__(self, size=64*64*3):
        super().__init__()
        self.size = size
        self.linear = FFN(size*2, size)
    def forward(self, x):
        B, T, C = x.shape
        hiddens = []
        h = torch.zeros(B, 1, C, device=x.device)
        for i in x.split(1, 1):
            h = self.linear(torch.cat([h,i], dim=-1)).split(C, -1)[0]
            hiddens.append(h)
        return torch.cat(hiddens, dim=1)
class PositionalEmbeddingsNoCaching(torch.nn.Module):
    def __init__(self, size=64*64*3):
        super().__init__()
        self.size = size
        inv_freq = 1. / (10000 ** (torch.arange(0, self.size, 2).float() / self.size))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        bs, t, c = x.shape
        pos = torch.arange(t, device=x.device).type(self.inv_freq.type())
        sinp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat([sinp.sin(), sinp.cos()], dim=-1)
        embd = torch.zeros((t, self.size), device=x.device).type(x.type())
        embd[:, :self.size] = emb
        
        return embd[None, :, :c].repeat(bs, 1, 1) * 0.001 + x

class AttentionPlusFFN(torch.nn.Module):
    def __init__(self, size=64*64*3, nheads=4):
        super().__init__()
        self.attn = SelfAttention(size, nheads)
        self.ffn = FFN(size)
    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

class NoiseInjection(torch.nn.Module):
    def __init__(self, size=64*64*3, nheads=4):
        super().__init__()
        self.attn = AttentionPlusFFN(size, nheads)
        self.attn2 = AttentionPlusFFN(size, nheads)
    def forward(self, x, temp):
        mu, logvar = self.attn(x), self.attn2(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (1-temp) * (mu) +  (temp) * (mu + eps * std), -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

class Transformer(torch.nn.Module):
    def __init__(self, nheads=4, size=64*64*3, nlayers=6):
        super().__init__()
        self.layers = []
        self.size = size
        self.noise_for_sampling = NoiseInjection(size, nheads)
        self.layers.append(RecurrentPositionalEmbedddings(size))
        for i in range(nlayers):
            self.layers.append(AttentionPlusFFN(size, nheads))
        self.layers = torch.nn.ModuleList(self.layers)
    def forward(self, x:torch.Tensor, temp=1.0):
        for i in range(len(self.layers)):
            if i == len(self.layers) // 2:
                x, sampling_loss = self.noise_for_sampling(x, temp)
            if i == 0:
                x = self.layers[i](x)
            elif i == 1:
                x = self.layers[i](x)
            else:
                x = torch.utils.checkpoint.checkpoint(self.layers[i], x)
        return torch.sigmoid(x), sampling_loss