# 从上面代码中看到，在tensor.storage()不变的情况下，可以通过调整stride来得到不同的tensor

# 例子中a, b的is_contiguous()为True，而c, d的is_contiguous()为False，因为：

# a, b展开后是[0,1,2,3,4,5,6,7,8,9,10,11]（与storage中的顺序一致）
# c, d展开后是[0,3,6,9,1,4,7,10,2,5,8,11]（与storage中的顺序不一致）

# c, d值完全相同，说明as_strided可以达到比view、permute更灵活的效果

import torch

def inspect(tensors):
    for t in tensors:
        print('-' * 10)
        print(t)
        print('is_contiguous: ', t.is_contiguous())
        print('stride: ', t.stride())
        print('storage ptr: ', t.untyped_storage().data_ptr())

if __name__ == '__main__':
    a = torch.arange(12)
    b = torch.as_strided(a, size=(3, 4), stride=(4, 1))
    c = torch.as_strided(a, size=(3, 4), stride=(1, 3))
    d = a.view(4, 3).permute(1, 0)
    inspect([a, b, c, d])


# ----------
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# is_contiguous:  True
# stride:  (1,)
# storage ptr:  93993499202944
# ----------
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# is_contiguous:  True
# stride:  (4, 1)
# storage ptr:  93993499202944
# ----------
# tensor([[ 0,  3,  6,  9],
#         [ 1,  4,  7, 10],
#         [ 2,  5,  8, 11]])
# is_contiguous:  False
# stride:  (1, 3)
# storage ptr:  93993499202944
# ----------
# tensor([[ 0,  3,  6,  9],
#         [ 1,  4,  7, 10],
#         [ 2,  5,  8, 11]])
# is_contiguous:  False
# stride:  (1, 3)
# storage ptr:  93993499202944