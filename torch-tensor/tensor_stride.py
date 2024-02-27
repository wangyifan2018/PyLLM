# tensor.stride()、torch.as_strided()


# 因为storage是顺序保存的，为了表示多维tensor，我们用size来定义tensor的每个维度尺寸，而用stride来定义从storage中取值的间隔步长

# 对于size=(m, n, p, q)的tensor，其默认的stride()=(n*p*q, p*q, q*1, 1)

# 当storage_offset设置为0时，首元素不变，因此tensor.data_ptr() == tensor.storage().data_ptr()

# 当storage_offset设置为非0时，首元素改变，因此tensor.data_ptr() != tensor.storage().data_ptr()


import torch

def inspect(tensors):
    print('=' * 30)
    for t in tensors:
        print('-' * 10)
        print(t)
        print(t.stride())
        print(t.data_ptr())
        print(t.untyped_storage().data_ptr())

if __name__ == '__main__':
  a = torch.randn([2, 3, 4, 5])
  print(a.stride())

  a = torch.arange(10)
  b = a[1:]
  inspect([a, b])

  b = torch.as_strided(a, size=[3, 3], stride=[3, 1], storage_offset=0)
  inspect([a, b])

  b = torch.as_strided(a, size=[3, 3], stride=[3, 1], storage_offset=1)
  inspect([a, b])

  b = torch.as_strided(a, size=[3, 3], stride=[1, 2], storage_offset=0)
  inspect([a, b])


# (60, 20, 5, 1)
# ==============================
# ----------
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# (1,)
# 94185115709120
# 94185115709120
# ----------
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
# (1,)
# 94185115709128
# 94185115709120
# ==============================
# ----------
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# (1,)
# 94185115709120
# 94185115709120
# ----------
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])
# (3, 1)
# 94185115709120
# 94185115709120
# ==============================
# ----------
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# (1,)
# 94185115709120
# 94185115709120
# ----------
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
# (3, 1)
# 94185115709128
# 94185115709120
# ==============================
# ----------
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# (1,)
# 94185115709120
# 94185115709120
# ----------
# tensor([[0, 2, 4],
#         [1, 3, 5],
#         [2, 4, 6]])
# (1, 2)
# 94185115709120
# 94185115709120