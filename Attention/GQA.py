import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_size, head_num, group_num):
        super(GroupQueryAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.group_num = group_num
        assert hidden_size % head_num == 0, "hidden_size must be divisible by head_num"
        assert head_num % group_num == 0, "head_num must be divisible by group_num"
        self.head_size = hidden_size // head_num

        self.ln1 = nn.LayerNorm(hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, self.group_num * self.head_size)
        self.W_v = nn.Linear(hidden_size, self.group_num * self.head_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.dropout1 = nn.Dropout(0.1)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.linear_up = nn.Linear(hidden_size, hidden_size * 4)
        self.gelu = nn.GELU()
        self.linear_down = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        assert hidden_size == self.hidden_size, "Input hidden size must match the initialized hidden size"

        # Multi-Query Attention
        x = x + self.ln1(x)
        Q = self.W_q(x).view(batch_size, seq_len, self.head_num, self.head_size).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.group_num, self.head_size).transpose(1, 2)
        K = torch.repeat_interleave(K, self.head_num // self.group_num, dim=1)
        V = self.W_v(x).view(batch_size, seq_len, self.group_num, self.head_size).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout1(attn_weights)
        V = torch.repeat_interleave(V, self.head_num // self.group_num, dim=1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout2(attn_output)

        # Feed Forward Network
        x =  x + self.ln2(attn_output)
        x = self.linear_up(x)
        x = self.gelu(x)
        x = self.linear_down(x)
        x = self.dropout2(x)
        return x


batch_size = 2
seq_len = 1024
hidden_size = 4096
num_heads = 8
group_num = 2

input_tensor = torch.randn(batch_size, seq_len, hidden_size)
my_attention = GroupQueryAttention(hidden_size, num_heads, group_num)
output_tensor = my_attention(input_tensor)
print(output_tensor.shape)

