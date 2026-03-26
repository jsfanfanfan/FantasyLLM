import torch
import torch.nn as nn
import torch.nn.functional as F

class flash_attn1(nn.Module):
    def __init__(self, Q):
        super().__init__()
        self.Tr, self.Tc = 4, 4     # Block Num
        self.seq_len = Q.size(0)
        self.hidden_size = Q.size(1)
        self.Br, self.Bc = self.seq_len // self.Tr, self.seq_len // self.Tc     # Block Size
    
    def forward(self, Q, K, V):
        O = torch.zeros((self.seq_len, self.hidden_size))
        cur_max = torch.full((self.seq_len,), float('-inf')) # 每一行的最大值
        cur_sum = torch.zeros((self.seq_len,)) # 每一行的当前和
        for j in range(self.Tc):    # outer loop: K,V
            Kj, Vj = K[j * self.Bc:(j + 1) * self.Bc,:], V[j * self.Bc:(j + 1) * self.Bc,:]  # from HBM load KV chunk
            for i in range(self.Tr):    # inner loop: Q, O
                Qi, Oi = Q[i * self.Br:(i + 1) * self.Br,:], O[i * self.Br:(i + 1) * self.Br,:]       # from HBM load QO chunk
                m_i, l_i = cur_max[i * self.Br:(i + 1) * self.Br], cur_sum[i * self.Br:(i + 1) * self.Br]   # from HBM load m,j chunk
                Sij = torch.matmul(Qi, Kj.transpose(-2,-1))      # (Br,h)*(h,Bc) = (Br,Bc)
                m_ij = Sij.max(dim=-1).values      # (Br,)
                Pij = torch.exp(Sij - m_ij.unsqueeze(-1))      # (Br, Bc) - (Br, 1) = (Br,Bc)
                l_ij = Pij.sum(dim=-1)           # (Br,)
                m_i_new = torch.max(m_i, m_ij)     # (Br,)
                l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij     # (Br,)
                Oi = (l_i / l_i_new).unsqueeze(-1) * torch.exp(m_i-m_i_new).unsqueeze(-1) * Oi + \
                    (torch.exp(m_ij-m_i_new)).unsqueeze(-1) * torch.matmul(Pij, Vj) / l_i_new.unsqueeze(-1)
                O[i * self.Br:(i + 1) * self.Br,:] = Oi   # write Oj to HBM
                cur_max[i * self.Br:(i + 1) * self.Br], cur_sum[i * self.Br:(i + 1) * self.Br] = m_i_new, l_i_new     # write m_j, l_j to HBM
        return O             

                

seq_len = 256
hidden_size = 1024
Q = torch.randn((256, 1024))
K = torch.randn((256, 1024))
V = torch.randn((256, 1024))
S = torch.matmul(Q, K.transpose(-2, -1))
S = F.softmax(S, dim=-1)
output = torch.matmul(S, V)

flash_attn = flash_attn1(Q)
output1 = flash_attn(Q, K, V)
print(f"Max difference: {torch.max(torch.abs(output - output1))}")