import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, seq_len):
        super(MultiHeadLatentAttention, self).__init__()
        assert num_heads * head_dim == hidden_size, "wrong param"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.position_dim = self.head_dim // 2

        self.W_dq = nn.Linear(self.hidden_size, self.head_dim * 4 * 3)
        self.W_dkv = nn.Linear(self.hidden_size, self.head_dim * 4)
        self.W_uq = nn.Linear(self.head_dim * 4 * 3, self.hidden_size)
        self.W_uk = nn.Linear(self.head_dim * 4, self.hidden_size)
        self.W_uv = nn.Linear(self.head_dim * 4, self.hidden_size)
        self.pos_q = nn.Linear(self.head_dim * 4 * 3, self.num_heads * head_dim // 2)
        self.pos_k = nn.Linear(self.hidden_size, self.head_dim // 2)
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.W_o = nn.Linear(hidden_size, hidden_size)



    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        Q_latent = self.W_dq(x)
        KV_latent = self.W_dkv(x)
        Q_content = self.W_uq(Q_latent).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q_rope = self.pos_q(Q_latent).reshape(batch_size, seq_len, self.num_heads, self.head_dim // 2)
        Q = torch.concat([Q_content, Q_rope], dim=-1).transpose(1, 2)
        K_content = self.W_uk(KV_latent).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K_rope = self.pos_k(x).unsqueeze(-2)
        K_rope = torch.repeat_interleave(K_rope,self.num_heads, -2)
        K = torch.concat([K_content, K_rope], dim=-1).transpose(1, 2)
        V = self.W_uv(KV_latent).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout1(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout2(attn_output)

        return attn_output


batch_size = 4
seq_len = 1024
hidden_size = 8192
num_heads = 64
head_dim = 128

input_tensor = torch.randn(batch_size, seq_len, hidden_size)
my_attention = MultiHeadLatentAttention(hidden_size, num_heads, head_dim, seq_len)
output_tensor = my_attention(input_tensor)
print(output_tensor.shape)
