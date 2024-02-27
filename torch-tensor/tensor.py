# tensor.storage().data_ptr()返回storage的地址，而tensor.data_ptr()返回首元素的地址。通过以下代码进行验证，可以看出
# a，b的storage().data_ptr()指向相同地址，修改b的值，a的值同步变化
# a的首元素是storage()的第一个值，b的首元素是storage的第二个值，c的首元素是storage的第二个值，所以
# a.data_ptr() == a.storage().data_ptr()
# b.data_ptr() == c.data_ptr()
# b.data_ptr() != b.storage().data_ptr()

import torch

def inspect(tensors):
    for t in tensors:
        print('-' * 10)
        print(t)
        print(t.data_ptr())
        print(t.untyped_storage().data_ptr())

if __name__ == '__main__':
    a = torch.arange(10)
    b = a[1:]
    b[1] = 100
    c = torch.as_strided(a, size=[9,], stride=[1,], storage_offset=1)
    inspect([a, b, c])


# ----------
# tensor([  0,   1, 100,   3,   4,   5,   6,   7,   8,   9])
# 94288912032704
# 94288912032704
# ----------
# tensor([  1, 100,   3,   4,   5,   6,   7,   8,   9])
# 94288912032712
# 94288912032704
# ----------
# tensor([  1, 100,   3,   4,   5,   6,   7,   8,   9])
# 94288912032712
# 94288912032704



