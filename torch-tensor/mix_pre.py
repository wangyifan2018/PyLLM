# torch可以在fp16，fp32间进行混合计算，如果包含混合类型，结果会转换成fp32
# torch可以在cpu、gpu间进行混合计算（仅限cpu上的标量），如果包含混合设备，结果会转移到gpu

import torch

def exp(device):
    if device in ('gpu', 'mix') and not torch.cuda.is_available():
        return
    if device == 'gpu':
        a0 = torch.tensor(3., device='cuda', requires_grad=True)
        b0 = torch.tensor(4., device='cuda', requires_grad=True, dtype=torch.float16)
        c0 = torch.tensor(4., device='cuda', requires_grad=True, dtype=torch.bfloat16)
    elif device == 'mix':
        a0 = torch.tensor(3., device='cpu', requires_grad=True)
        b0 = torch.rand((2, 2), device='cuda', requires_grad=True, dtype=torch.float16)
        c0 = torch.rand((2, 2), device='cuda', requires_grad=True, dtype=torch.bfloat16)
    else:
        a0 = torch.tensor(3., device='cpu', requires_grad=True)
        b0 = torch.tensor(4., device='cpu', requires_grad=True, dtype=torch.float16)
        c0 = torch.tensor(4., device='cpu', requires_grad=True, dtype=torch.bfloat16)
    ab = a0 * b0 + a0 + b0
    ac = a0 * c0 + a0 + c0
    bc = b0 * c0 + b0 + c0
    for name, param in zip(('a0', 'b0', 'c0', 'ab', 'ac', 'bc'), (a0, b0, c0, ab, ac, bc)):
        print(name, ' | ', param.is_leaf, ' | ', param.requires_grad, ' | ', param.dtype, ' | ', param.device)

if __name__ == "__main__":
    print('=' * 10, 'cpu', '=' * 10)
    exp('cpu')
    print('=' * 10, 'gpu', '=' * 10)
    exp('gpu')
    print('=' * 10, 'mix', '=' * 10)
    exp('mix')


# ========== cpu ==========
# a0  |  True  |  True  |  torch.float32  |  cpu
# b0  |  True  |  True  |  torch.float16  |  cpu
# c0  |  True  |  True  |  torch.bfloat16  |  cpu
# ab  |  False  |  True  |  torch.float32  |  cpu
# ac  |  False  |  True  |  torch.float32  |  cpu
# bc  |  False  |  True  |  torch.float32  |  cpu
# ========== gpu ==========
# a0  |  True  |  True  |  torch.float32  |  cuda:0
# b0  |  True  |  True  |  torch.float16  |  cuda:0
# c0  |  True  |  True  |  torch.bfloat16  |  cuda:0
# ab  |  False  |  True  |  torch.float32  |  cuda:0
# ac  |  False  |  True  |  torch.float32  |  cuda:0
# bc  |  False  |  True  |  torch.float32  |  cuda:0
# ========== mix ==========
# a0  |  True  |  True  |  torch.float32  |  cpu
# b0  |  True  |  True  |  torch.float16  |  cuda:0
# c0  |  True  |  True  |  torch.bfloat16  |  cuda:0
# ab  |  False  |  True  |  torch.float16  |  cuda:0
# ac  |  False  |  True  |  torch.bfloat16  |  cuda:0
# bc  |  False  |  True  |  torch.float32  |  cuda:0