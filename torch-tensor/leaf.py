# 直接复制（如b=a），不会新增节点，所以所有属性不变
# 其他符号运算、to运算（类型转换，device转换等），都会在计算图中创建新的节点，新节点的is_leaf等于False


import torch

a0 = torch.tensor(3., requires_grad=True)
a1 = a0
a2 = a0 * 1
a3 = a0.to(torch.float16)
a4 = a0.cuda() if torch.cuda.is_available() else a0.cpu()

for name, param in zip(('a0', 'a1', 'a2', 'a3', 'a4'), (a0, a1, a2, a3, a4)):
    print(name, ' | ', param.is_leaf, ' | ', param.requires_grad)

# ### 输出 ###
# a0  |  True  |  True
# a1  |  True  |  True
# a2  |  False  |  True
# a3  |  False  |  True
# a4  |  False  |  True