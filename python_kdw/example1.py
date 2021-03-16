import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell)
print(my_cell(x, h))
print(traced_cell(x, h))

print('\n\n')
print(traced_cell.graph)

print('\n\n')
print(traced_cell.code)

