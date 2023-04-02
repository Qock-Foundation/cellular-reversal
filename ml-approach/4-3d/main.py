import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
device = 'cuda'

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    n_channels = 256 #64
    self.stack = nn.Sequential(
      nn.Conv3d(in_channels=3, out_channels=n_channels, kernel_size=3, padding=1),
      nn.BatchNorm3d(num_features=n_channels),
      nn.ReLU(),
      nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1),
      nn.BatchNorm3d(num_features=n_channels),
      nn.ReLU(),
      nn.Conv3d(in_channels=n_channels, out_channels=3, kernel_size=3, padding=1),
      nn.BatchNorm3d(num_features=3),
      nn.ReLU()
    )
  def forward(self, x):
    return x + self.stack(x)
model = NN().to(device)
print(sum(p.numel() for p in model.parameters()), 'parameters')

flogs = open('logs', 'w')

def show_tensor(tensor, filename):
  arr = tensor.detach().cpu().numpy()
  arr = (arr - arr.min()) / (arr.max() - arr.min())
  fig, axes = plt.subplots(4, 6, figsize=(18, 12))
  for i in range(4):
    for j in range(6):
      axes[i][j].imshow(arr[:, i * 6 + j].transpose(1, 2, 0))
  plt.savefig(filename)
  plt.clf()
  plt.close()

cnt, n, r = 16384, 24, 8 #44, 12
input_images = torch.zeros(cnt, 3, n, n, n)
output_images = torch.zeros(cnt, 3, n, n, n)
numbers = torch.randn(cnt, 3, (2 * r + 1) // 3)
input_images[:, :, n // 2, n // 2 - 1, n // 2 - r:n // 2 + r] = 1
input_images[:, :, n // 2, n // 2, n // 2 - r:n // 2 + r:3] = -1
for i in range(2):
  for j in range(2):
    for k in range(2):
      input_images[:, :, n // 2 + i, n // 2 + j, n // 2 - r + 1 + k:n // 2 + r:3] = numbers
#output_images[:, :, :, 1:, :] = input_images[:, :, :, :-1, :]
output_images[:, :, n // 2, n // 2 - r:n // 2 + r, n // 2 - 1] = 1
output_images[:, :, n // 2, n // 2 - r:n // 2 + r:3, n // 2] = -1
for i in range(2):
  for j in range(2):
    for k in range(2):
      output_images[:, :, n // 2 + i, n // 2 - r + 1 + j:n // 2 + r:3, n // 2 + k] = numbers
show_tensor(input_images[0], 'sample_input.png')
show_tensor(output_images[0], 'sample_output.png')

batch_size = 4 #70
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  #1e-7 min
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)  #0.8
for epoch in range(10000):
  epoch_loss = 0
  for batch_i in range(100):
    optimizer.zero_grad()
    inds = torch.randint(cnt, size=(batch_size,))
    input = input_images[inds].to(device).requires_grad_()
    x = input
    for t in range(100): #300
      x = model(x)
    rect_i_from, rect_i_to = n // 2, n // 2 + 2
    rect_j_from, rect_j_to = n // 2 - r, n // 2 + r
    rect_k_from, rect_k_to = n // 2 - 1, n // 2 + 2
    #rect_i_from, rect_i_to = 0, n
    #rect_j_from, rect_j_to = 0, n
    #rect_k_from, rect_k_to = 0, n
    output = x[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to, rect_k_from:rect_k_to]
    target = output_images[inds].to(device)[:, :, rect_i_from:rect_i_to, rect_j_from:rect_j_to, rect_k_from:rect_k_to]
    loss = criterion(output, target)
    epoch_loss += loss.item()
    loss.backward()
    flogs.write('before clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write(f'input {torch.abs(input).mean().item()} output {torch.abs(x).mean().item()} input.grad {torch.abs(input.grad).mean().item()}\n')
    nn.utils.clip_grad_norm_(model.parameters(), 1e-7)
    flogs.write('after clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write('\n')
    optimizer.step()
  epoch_loss /= 100
  scheduler.step()
  print('epoch', epoch, 'loss', epoch_loss)
  flogs.write(f'epoch {epoch} loss {epoch_loss}\n\n')
  show_tensor(x[0], str(epoch).zfill(4) + '.png')
