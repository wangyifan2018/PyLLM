# 计算图中可以包括不同精度、不同device的tensor

# backward之后grad的精度和device，和forward时保持一致

# tensor.to()操作（dtype或device转换）的grad_fn都是<ToCopyBackward0>，说明类型转换、设备转换在pytorch中都是当做同一个可导的操作

import torch

def exp(device, backward_fn='d.backward'):
    if device in ('gpu', 'mix') and not torch.cuda.is_available(): return

    a_fp32 = torch.tensor(3., device='cuda', requires_grad=True) if device == 'gpu' else torch.tensor(3., requires_grad=True)
    a_fp16 = a_fp32.to(torch.float16)
    if device == 'mix':
        a_fp16 = a_fp16.cuda()
    b_fp16 = torch.tensor(4., device='cuda', requires_grad=True, dtype=torch.bfloat16) if device in ('gpu', 'mix') else \
        torch.tensor(4., requires_grad=True, dtype=torch.bfloat16)

    c1 = (2 * a_fp16 + 5 * b_fp16) ** 2
    c2 = (3 * a_fp32 + 7 * b_fp16) ** 2

    d = 8 * c1 + 9 * c2

    params = (a_fp32, a_fp16, b_fp16, c1, c2, d)
    names = ('a_fp32', 'a_fp16', 'b_fp16', 'c1', 'c2', 'd')
    for param in params:
        param.retain_grad()

    print('=' * 5, backward_fn)
    eval(backward_fn)()
    for i, param in enumerate(params):
        print(names[i], ': ', param.is_leaf, ' | ', param, ' | ', param.grad)

if __name__ == "__main__":
    print('=' * 30, 'cpu', '=' * 30)
    exp('cpu')
    print('=' * 30, 'gpu', '=' * 30)
    exp('gpu')
    print('=' * 30, 'mix', '=' * 30)
    exp('mix')


#     ============================== cpu ==============================
# ===== d.backward
# a_fp32 :  True  |  tensor(3., requires_grad=True)  |  tensor(2830.)
# a_fp16 :  False  |  tensor(3., dtype=torch.float16, grad_fn=<ToCopyBackward0>)  |  tensor(832., dtype=torch.float16)
# b_fp16 :  True  |  tensor(4., dtype=torch.bfloat16, requires_grad=True)  |  tensor(6720., dtype=torch.bfloat16)
# c1 :  False  |  tensor(676., grad_fn=<PowBackward0>)  |  tensor(8.)
# c2 :  False  |  tensor(1369., grad_fn=<PowBackward0>)  |  tensor(9.)
# d :  False  |  tensor(17729., grad_fn=<AddBackward0>)  |  tensor(1.)
# ============================== gpu ==============================
# ===== d.backward
# a_fp32 :  True  |  tensor(3., device='cuda:0', requires_grad=True)  |  tensor(2830., device='cuda:0')
# a_fp16 :  False  |  tensor(3., device='cuda:0', dtype=torch.float16, grad_fn=<ToCopyBackward0>)  |  tensor(832., device='cuda:0', dtype=torch.float16)
# b_fp16 :  True  |  tensor(4., device='cuda:0', dtype=torch.bfloat16, requires_grad=True)  |  tensor(6720., device='cuda:0', dtype=torch.bfloat16)
# c1 :  False  |  tensor(676., device='cuda:0', grad_fn=<PowBackward0>)  |  tensor(8., device='cuda:0')
# c2 :  False  |  tensor(1369., device='cuda:0', grad_fn=<PowBackward0>)  |  tensor(9., device='cuda:0')
# d :  False  |  tensor(17729., device='cuda:0', grad_fn=<AddBackward0>)  |  tensor(1., device='cuda:0')
# ============================== mix ==============================
# ===== d.backward
# a_fp32 :  True  |  tensor(3., requires_grad=True)  |  tensor(2830.)
# a_fp16 :  False  |  tensor(3., device='cuda:0', dtype=torch.float16, grad_fn=<ToCopyBackward0>)  |  tensor(832., device='cuda:0', dtype=torch.float16)
# b_fp16 :  True  |  tensor(4., device='cuda:0', dtype=torch.bfloat16, requires_grad=True)  |  tensor(6720., device='cuda:0', dtype=torch.bfloat16)
# c1 :  False  |  tensor(676., device='cuda:0', grad_fn=<PowBackward0>)  |  tensor(8., device='cuda:0')
# c2 :  False  |  tensor(1369., device='cuda:0', grad_fn=<PowBackward0>)  |  tensor(9., device='cuda:0')
# d :  False  |  tensor(17729., device='cuda:0', grad_fn=<AddBackward0>)  |  tensor(1., device='cuda:0')