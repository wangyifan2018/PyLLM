# 我们看到tensor.cuda()操作会在计算图中增加一个节点。我们平时在用gpu训练模型时经常使用model.cuda()，是否也会在计算图中增加一个节点呢。

# model.cuda()操作并不是简单的调用tensor.to()，而是重新创建了一个Parameter，因为计算图是调用model.forward()之后才创建，所以在forward之前修改值并不会影响反向传播

# retrain_grad：默认情况下backward过程中，非叶子节点的梯度使用之后就会直接释放，所以查看飞叶子节点的grad为None，如果不想释放非叶子节点的梯度，可以在backward之前调用tensor.retain_grad()


import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def exp(convertor='official'):
    if not torch.cuda.is_available(): return
    torch.manual_seed(1)

    model = MyModel()
    for name, param in model.named_parameters():
        print(name, '|', param.device, '|', param.is_leaf, '|', param.dtype)

    if convertor == 'official':
        model = model.cuda()
    else:
        ############### 模拟model.cuda()操作 ###############
        def convert(module, fn):
            for module in module.children():
                convert(module, fn)
            for key, param in module._parameters.items():
                if param is not None:
                    with torch.no_grad():
                        param_applied = fn(param)
                    module._parameters[key] = torch.nn.Parameter(param_applied, param.requires_grad)
            for key, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[key] = fn(buf)

        convert(model, fn=lambda x: x.cuda())

    for name, param in model.named_parameters():
        param.retain_grad()

    x = torch.randn((2, 4), device='cuda')
    y = model(x)
    y.retain_grad()
    torch.sum(y).backward()

    print('-' * 10)
    for name, param in model.named_parameters():
        print(name, '|', param.device, '|', param.is_leaf, '|', param.dtype, '|', param.grad)

    print('-' * 10)
    for param in (x, y):
        print(param.device, '|', param.is_leaf, '|', param.dtype, '|', param.grad)

if __name__ == "__main__":
    print('=' * 30, 'official', '=' * 30)
    exp('official')
    print('=' * 30, 'custom', '=' * 30)
    exp('custom')


#     ============================== official ==============================
# linear1.weight | cpu | True | torch.float32
# linear1.bias | cpu | True | torch.float32
# linear2.weight | cpu | True | torch.float32
# linear2.bias | cpu | True | torch.float32
# ----------
# linear1.weight | cuda:0 | True | torch.float32 | tensor([[ 1.9883e-03, -8.2887e-01,  5.5231e-01, -5.2989e-01],
#         [ 1.5958e-03, -6.6524e-01,  4.4328e-01, -4.2529e-01],
#         [-1.2297e-03,  5.1263e-01, -3.4159e-01,  3.2772e-01],
#         [-3.2198e-04,  1.3422e-01, -8.9437e-02,  8.5807e-02]], device='cuda:0')
# linear1.bias | cuda:0 | True | torch.float32 | tensor([-0.6929, -0.5561,  0.4285,  0.1122], device='cuda:0')
# linear2.weight | cuda:0 | True | torch.float32 | tensor([[ 0.1289,  1.2692,  0.5439, -0.1668],
#         [ 0.1289,  1.2692,  0.5439, -0.1668]], device='cuda:0')
# linear2.bias | cuda:0 | True | torch.float32 | tensor([2., 2.], device='cuda:0')
# ----------
# cuda:0 | True | torch.float32 | None
# cuda:0 | False | torch.float32 | tensor([[1., 1.],
#         [1., 1.]], device='cuda:0')
# ============================== custom ==============================
# linear1.weight | cpu | True | torch.float32
# linear1.bias | cpu | True | torch.float32
# linear2.weight | cpu | True | torch.float32
# linear2.bias | cpu | True | torch.float32
# ----------
# linear1.weight | cuda:0 | True | torch.float32 | tensor([[ 1.9883e-03, -8.2887e-01,  5.5231e-01, -5.2989e-01],
#         [ 1.5958e-03, -6.6524e-01,  4.4328e-01, -4.2529e-01],
#         [-1.2297e-03,  5.1263e-01, -3.4159e-01,  3.2772e-01],
#         [-3.2198e-04,  1.3422e-01, -8.9437e-02,  8.5807e-02]], device='cuda:0')
# linear1.bias | cuda:0 | True | torch.float32 | tensor([-0.6929, -0.5561,  0.4285,  0.1122], device='cuda:0')
# linear2.weight | cuda:0 | True | torch.float32 | tensor([[ 0.1289,  1.2692,  0.5439, -0.1668],
#         [ 0.1289,  1.2692,  0.5439, -0.1668]], device='cuda:0')
# linear2.bias | cuda:0 | True | torch.float32 | tensor([2., 2.], device='cuda:0')
# ----------
# cuda:0 | True | torch.float32 | None
# cuda:0 | False | torch.float32 | tensor([[1., 1.],
#         [1., 1.]], device='cuda:0')