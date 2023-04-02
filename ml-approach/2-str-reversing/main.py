import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
device = 'cuda'

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    n_channels = 1024 #128
    self.stack = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=3),
      nn.ReLU()
    )
  def forward(self, x):
    return x + self.stack(x)
model = NN().to(device)
print(sum(p.numel() for p in model.parameters()), 'parameters')

flogs = open('logs', 'w')

def show_tensor(tensor, filename):
  plt.imshow(tensor.detach().cpu().numpy().transpose(1, 2, 0))
  plt.savefig(filename)
  plt.clf()

cnt, n, r = 524288, 24, 8 #44, 12
input_images = torch.zeros(cnt, 3, n, n)
output_images = torch.zeros(cnt, 3, n, n)
numbers = np.random.randn(cnt, 3, 2 * r)
input_images[:, :, n // 2, n // 2 - r:n // 2 + r] = torch.tensor(numbers)
#output_images[:, :, n // 2, n // 2 - r:n // 2 + r] = torch.tensor(numbers[:, :, ::-1].copy())
output_images[:, :, n // 2 - r:n // 2 + r, n // 2] = torch.tensor(numbers)
show_tensor(input_images[0], 'sample_input.png')
show_tensor(output_images[0], 'sample_output.png')

batch_size = 16
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #4
for epoch in range(10000):
  epoch_loss = 0
  for batch_i in range(100):
    optimizer.zero_grad()
    inds = torch.randint(cnt, size=(batch_size,))
    input = input_images[inds].to(device).requires_grad_()
    x = input
    for t in range(100): #300
      x = model(x)
    rect_i_from, rect_i_to, rect_j_from, rect_j_to = n // 2 - r, n // 2 + r, n // 2, n // 2 + 1
    output = x[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to]
    target = output_images[inds].to(device)[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to]
    loss = criterion(output, target)
    epoch_loss += loss.item()
    loss.backward()
    flogs.write('before clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write(f'input {torch.abs(input).mean().item()} output {torch.abs(x).mean().item()} input.grad {torch.abs(input.grad).mean().item()}\n')
    nn.utils.clip_grad_norm_(model.parameters(), 1) #value
    flogs.write('after clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write('\n')
    optimizer.step()
  epoch_loss /= 100
  print('epoch', epoch, 'loss', epoch_loss)
  flogs.write(f'epoch {epoch} loss {epoch_loss}\n\n')
  show_tensor(x[0], str(epoch).zfill(4) + '.png')
