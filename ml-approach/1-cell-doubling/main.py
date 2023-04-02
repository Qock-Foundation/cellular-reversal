import torch, torch.nn as nn
import matplotlib.pyplot as plt
device = 'cuda'

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    n_channels = 128
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

cnt, n, r = 524288, 64, 8
input_images = torch.zeros(cnt, 3, n, n)
output_images = torch.zeros(cnt, 3, n, n)
for di in range(-r, r + 1):
  for dj in range(-r, r + 1):
    if di * di + dj * dj <= r * r:
      numbers = torch.randn(cnt, 3)
      input_images[:, :, n // 2 + di, n // 2 + dj] = numbers
      #output_images[:, :, n // 2 + di, n // 4 + dj] = numbers
      #output_images[:, :, n // 2 + di, 3 * n // 4 + dj] = numbers
for i in range(0, n - 2, 2):
  for j in range(0, n - 2, 2):
#    input_images[:, :, i + 1, j] = (input_images[:, :, i, j] + input_images[:, :, i + 2, j]) / 2
#    input_images[:, :, i, j + 1] = (input_images[:, :, i, j] + input_images[:, :, i, j + 2]) / 2
#    input_images[:, :, i + 1, j + 1] = (input_images[:, :, i, j] + input_images[:, :, i + 2, j]
#                                        + input_images[:, :, i, j + 2] + input_images[:, :, i + 2, j + 2]) / 4
    input_images[:, :, i, j + 1] = input_images[:, :, i, j]
    input_images[:, :, i + 1, j] = input_images[:, :, i, j]
    input_images[:, :, i + 1, j + 1] = input_images[:, :, i, j]
output_images[:, :, n // 2 - r:n // 2 + r, n // 4 - r:n // 4 + r] = input_images[:, :, n // 2 - r:n // 2 + r, n // 2 - r:n // 2 + r]
output_images[:, :, n // 2 - r:n // 2 + r, 3 * n // 4 - r:3 * n // 4 + r] = input_images[:, :, n // 2 - r:n // 2 + r, n // 2 - r:n // 2 + r]
show_tensor(input_images[0], 'sample_input.png')
show_tensor(output_images[0], 'sample_output.png')

batch_size = 16
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1000):
  epoch_loss = 0
  for batch_i in range(100):
    optimizer.zero_grad()
    inds = torch.randint(cnt, size=(batch_size,))
    input = input_images[inds].to(device).requires_grad_()
    x = input
    for t in range(100):
      x = model(x)
    rect_i_from, rect_i_to, rect_j_from, rect_j_to = 0, n, 0, n  #n // 2 - r, n // 2 + r + 1, n // 4 - r, 3 * n // 4 + r + 1
    output = x[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to]
    target = output_images[inds].to(device)[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to]
    loss = criterion(output, target)
    epoch_loss += loss.item()
    loss.backward()
    flogs.write('before clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write(f'input {torch.abs(input).mean().item()} output {torch.abs(x).mean().item()} input.grad {torch.abs(input.grad).mean().item()}\n')
    nn.utils.clip_grad_value_(model.parameters(), 1)
    flogs.write('after clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write('\n')
    optimizer.step()
  epoch_loss /= 100
  print('epoch', epoch, 'loss', epoch_loss)
  flogs.write(f'epoch {epoch} loss {epoch_loss}\n\n')
  show_tensor(x[0], str(epoch).zfill(4) + '.png')
