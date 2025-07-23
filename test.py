import math
import torch
 
# Query, Key 初始化
Q = torch.tensor([2.0, 3.0, 1.0])
K1 = torch.tensor([1.0, 2.0, 1.0])  # 'apple'
K2 = torch.tensor([1.0, 1.0, 2.0])  # 'orange'
 
# 点积计算
dot_product1 = torch.dot(Q, K1)
dot_product2 = torch.dot(Q, K2)
 
# 缩放因子
d_k = Q.size(0) # 一维张量，size（0）就行，Q和K大小一致，写哪个都行，本例=3
scale_factor = math.sqrt(d_k)
 
# 缩放点积
scaled_dot_product1 = dot_product1 / scale_factor
scaled_dot_product2 = dot_product2 / scale_factor


# Softmax 计算
weights = torch.nn.functional.softmax(torch.tensor([scaled_dot_product1, scaled_dot_product2]), dim=0)
 
print("权重:", weights)