import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
device = 'cuda'

wordlen = 5
fontsize = 4
n = (wordlen + 2) * fontsize

def show_tensor(tensor, filename):
  arr = tensor.detach().cpu().numpy()
  arr[arr > 1.5] = 1.5
  arr[arr < -1.5] = -1.5
  arr = (arr - arr.min()) / (arr.max() - arr.min())
  fig, axes = plt.subplots(5, 6, figsize=(18, 15))
  for i in range(5):
    for j in range(6):
      if i == 0 and j == 0 or i == 4 and j == 5:
        axes[i][j].axis('off')
        continue
      axes[i][j].imshow(arr[:, i * 6 + j - 1].transpose(1, 2, 0))
  plt.savefig(filename)
  plt.clf()
  plt.close()

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    n_channels = 128
    self.learnable_ground = nn.Parameter(torch.randn(3, n, n, n))
    self.letter_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=3*fontsize**3)  # num_embeddings also affects l. 64
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
  def forward(self, words, io_filenames=None, o_filename=None):
    cnt_words = len(words)
    device = self.letter_embeddings.weight.device
    words_inv = torch.tensor(words[:, ::-1].copy(), device=device)
    words = torch.tensor(words, device=device)
    fields = self.learnable_ground.repeat(len(words), 1, 1, 1, 1)
    word_il, word_ir = n // 2, n // 2 + fontsize
    word_j1l, word_j1r = fontsize, 2 * fontsize
    word_j2l, word_j2r = n - 2 * fontsize, n - fontsize
    word_kl, word_kr = fontsize, n - fontsize
    word_shape = 3, word_ir - word_il, word_j1r - word_j1l, word_kr - word_kl
    fields[:, :, word_il:word_ir, word_j1l:word_j1r, word_kl:word_kr] \
        = self.letter_embeddings(words).reshape(cnt_words, *word_shape)\
            .reshape(cnt_words, wordlen, 3, fontsize, fontsize, fontsize).permute(0, 2, 3, 4, 1, 5)\
            .reshape(cnt_words, 3, fontsize, fontsize, wordlen * fontsize)
    if io_filenames is not None:
      show_tensor(fields[0], io_filenames[0])
    for i in range(30):
      fields = fields + self.stack(fields)
    if o_filename is not None:
      show_tensor(fields[0], o_filename)
    target = torch.zeros(cnt_words, 3, n, n, n, device=device)
    target[:, :, word_il:word_ir, word_j2l:word_j2r, word_kl:word_kr] \
        = self.letter_embeddings(words_inv).reshape(cnt_words, *word_shape)\
            .reshape(cnt_words, wordlen, 3, fontsize, fontsize, fontsize).permute(0, 2, 3, 4, 1, 5)\
            .reshape(cnt_words, 3, fontsize, fontsize, wordlen * fontsize)
    if io_filenames is not None:
      show_tensor(target[0], io_filenames[1])
    letters_loss = ((self.letter_embeddings(torch.tensor(0, device=device)) - 1) ** 2).mean() \
                   + ((self.letter_embeddings(torch.tensor(1, device=device)) + 1) ** 2).mean()
    return fields[:, :, word_il:word_ir, word_j1l:word_j1r, word_kl:word_kr], \
           target[:, :, word_il:word_ir, word_j2l:word_j2r, word_kl:word_kr], letters_loss

model = NN().to(device)
print(sum(p.numel() for p in model.parameters()), 'parameters')

flogs = open('logs', 'w')

batch_size = 16
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  #1e-7 min
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)  #0.8
model(np.random.randint(2, size=(batch_size, wordlen)), ('sample_input.png', 'sample_output.png'))
for epoch in range(10000):
  epoch_loss = 0
  cnt_iters = 100
  for batch_i in range(cnt_iters):
    words = np.random.randint(2, size=(batch_size, wordlen))
    optimizer.zero_grad()
    output, target, ll = model(words, o_filename=str(epoch).zfill(4)+'.png' if batch_i==0 else None)
    loss = criterion(output, target) + ll
    epoch_loss += loss.item()
    loss.backward()
    flogs.write('before clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    flogs.write('after clipping:\n')
    for name, p in model.named_parameters():
      flogs.write(f'\t{name=}: {p.shape=} {torch.abs(p.grad).mean().item()=}\n')
    flogs.write('\n')
    optimizer.step()
  epoch_loss /= cnt_iters
  scheduler.step()
  print('epoch', epoch, 'loss', epoch_loss)
  flogs.write(f'epoch {epoch} loss {epoch_loss}\n\n')
