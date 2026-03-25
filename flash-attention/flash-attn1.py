import torch
import torch.nn as nn
import torch.nn.functional as F

class flash_attn1(nn.Module):
    def __init__(self, Q, K, V):
        super().__init__()
        self.Tr, self.Tc = 4, 4     # Block Num
        self.seq_len = Q.size(0)
        self.hidden_size = Q.size(1)
        self.Br, self.Bc = self.seq_len // self.Tr, self.seq_len // self.Tc     # Block Size
    
    def forward(self, Q, K, V):
        cur_max = torch.tensor([float('-inf')] * self.seq_len) # 每一行的最大值
        cur_sum = torch.tensor([0] * self.seq_len) # 每一行的当前和
        O = torch.full((self.seq_len, self.seq_len), float('-inf'))
        for i in range(self.Bc):    # outer loop: K,V
            Ki, Vi = K[i*self.Bc:(i+1)*self.Bc,:], V[i*self.Bc:(i+1)*self.Bc,:]  # from HBM load KV chunk
            for j in range(self.Br):    # inner loop: Q, O
                Qj,Oj = Q[j*self.Br:(j+1)*self.Br,:], O[j*self.Br:(j+1)*self.Br,i*self.Bc:(i+1)*self.Bc,:]       # from HBM load QO chunk
                m_j, l_j = cur_max[i * self.Br:(i+1) * self.Br], cur_sum[i * self.Br:(i+1) * self.Br]   # from HBM load m,j chunk
                Sij = Qj @ Ki.transpose(-2,-1)      # (Br,h)*(h,Bc) = (Br,Bc)
                row_max, row_sum = Sij.max(dim=-1).values, Sij.sum(dim=-1)
                Pij = torch.exp(Sij - row_max)      # (Br,Bc)
                new_max = max(old_max, row_max)
                new_sum = old_sum * torch.exp(old_max - new_max) + row_sum * torch.exp(row_max - new_max)

                cur_max[i * self.Br:(i+1) * self.Br], cur_sum[i * self.Br:(i+1) * self.Br] = new_max, new_sum




                

                

seq_len = 256
hidden_size = 1024
Q = torch.randn((256, 1024))
K = torch.randn((256, 1024))
V = torch.randn((256, 1024))
S = torch.matmul(Q, K.transpose(-2, -1))
S = F.softmax(S, dim=-1)
output = torch.matmul(S, V)

flash_attn = flash_attn1()
output1 = flash_attn(Q, K, V)
assert torch.allclose(output, output1)
