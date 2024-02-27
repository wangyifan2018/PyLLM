# 假定用M表示pytorch模型，用X表示输入（函数自变量），Y表示输出（函数因变量），那么有Y=M(X)。将M想象如下所示的有向无环图（包含节点、边）（或者树）  ()[grad_leaf.jpg]


import torch

def inspect(tensors):
    for t in tensors:
        print('-' * 10)
        print(t)
        print('requires_grad: ', t.requires_grad)
        print('is_leaf: ', t.is_leaf)

if __name__ == '__main__':
    x0 = torch.arange(4, dtype=torch.float, requires_grad=True)
    h0 = x0 * 10
    with torch.no_grad():
        h1 = x0 * x0
    h2 = h0 + h1
    h3 = h2.detach()
    h4 = h1 + h3
    inspect([x0, h0, h1, h2, h3, h4])


# ----------
# tensor([0., 1., 2., 3.], requires_grad=True)
# requires_grad:  True
# is_leaf:  True
# ----------
# tensor([ 0., 10., 20., 30.], grad_fn=<MulBackward0>)
# requires_grad:  True
# is_leaf:  False
# ----------
# tensor([0., 1., 4., 9.])
# requires_grad:  False
# is_leaf:  True
# ----------
# tensor([ 0., 11., 24., 39.], grad_fn=<AddBackward0>)
# requires_grad:  True
# is_leaf:  False
# ----------
# tensor([ 0., 11., 24., 39.])
# requires_grad:  False
# is_leaf:  True
# ----------
# tensor([ 0., 12., 28., 48.])
# requires_grad:  False
# is_leaf:  True

# is_leaf标记tensor是否属于叶子节点

# 其他requires_grad为False的节点，如通过torch.no_grad（上图h1）、detach（上图h3）、或由requires_grad全为False的上游节点计算得到（上图h4）

# requires_grad标记标记在backward时，是否需要计算该节点的梯度。根据链式求导原理，如要计算某非叶子节点对其他节点的梯度，需要确保计算链路上的节点requires_grad都为True


# 如上图中x0->(h0,h1)->h2是一个有效的正向计算，而h2->h0->x0是一个有效的反向计算，因为h1的requires_grad为False，所以反向计算时梯度不会经过h1传播



