import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class NativeSparseAttention(nn.Module):
    def __init__(self, hidden_size=512, head_num=8, comp_block=32, sel_block=64, win_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.comp_block = comp_block
        