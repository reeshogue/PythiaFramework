import torch
import math

class Attention(torch.nn.Module):
    #From Nystromformer: https://arxiv.org/abs/2102.03902
    def __init__(self, size, nlandmarks=8):
        super().__init__()
        self.size = size
        self.nlandmarks = nlandmarks
        
    def forward(self, q, k, v):
        B, T, C = q.shape
        q = q.reshape(B, C//self.size, T, self.size)
        k = k.reshape(B, C//self.size, T, self.size)
        k = k.reshape(B, C//self.size, T, self.size)
        v = v.reshape(B, C//self.size, T, self.size)

        if T < self.nlandmarks:
            attn = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.size), dim=-1), v)
            return attn.reshape(B, T, C)
 
        if T % self.nlandmarks == 0:
            q_landmarks = q.reshape(B, C//self.size, self.nlandmarks, T//self.nlandmarks, self.size).mean(dim=-2)
            k_landmarks = k.reshape(B, C//self.size, self.nlandmarks, T//self.nlandmarks, self.size).mean(dim=-2)
        else:
            q_temp = list(torch.split(q, 1, -2))
            for i in range(T % self.nlandmarks):
                q_temp.pop(0)
            q_temp = torch.cat(q_temp, -2)
            q_landmarks = q_temp.reshape(B, C//self.size, self.nlandmarks, T//self.nlandmarks, self.size).mean(dim=-2)
            
            k_temp = list(torch.split(k, 1, -2))
            for i in range(T % self.nlandmarks):
                k_temp.pop(0)
            k_temp = torch.cat(k_temp, -2)
            k_landmarks = k_temp.reshape(B, C//self.size, self.nlandmarks, T//self.nlandmarks, self.size).mean(dim=-2)


        kernel_1 = torch.softmax(torch.matmul(q, k_landmarks.transpose(-1,-2))/math.sqrt(self.size), dim=-1)
        kernel_2 = torch.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1,-2))/math.sqrt(self.size), dim=-1)
        kernel_3 = torch.softmax(torch.matmul(q_landmarks, k.transpose(-1,-2))/math.sqrt(self.size), dim=-1)
        out = torch.matmul(torch.matmul(kernel_1, self.iterative_pinv(kernel_2)), torch.matmul(kernel_3, v))
        return out.reshape(B, T, C)

    def iterative_pinv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1,-2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V
