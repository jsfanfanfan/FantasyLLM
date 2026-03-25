import math
import torch
import torch.nn.functional as F

def safe_online_softmax(x):
    cur_max = x[0]
    cur_sum = 1.0
    
    for i in range(1, len(x)):
        val = x[i]
        if val > cur_max:
            cur_sum = cur_sum * math.exp(cur_max - val) + 1
            cur_max = val
        else:
            cur_sum += math.exp(val - cur_max)
    
    result = []
    for val in x:
        result.append(math.exp(val - cur_max) / cur_sum)
    
    return result

x = [3.0, 2.0, 5.0, 1.0]
res = safe_online_softmax(x)
print(res, sum(res))

# PyTorch 结果
# torch_result = F.softmax(torch.tensor(x), dim=0)
# print("PyTorch结果:", [f"{r:.6f}" for r in torch_result.tolist()])
# print("PyTorch总和:", torch_result.sum().item())