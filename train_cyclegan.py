import torch
import torch.utils.data
import models_conv as models
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import eval
from utils import *

batch_size = 32
init_lr = 1e-4
total_iter = 100001
ckpt_iter = 1000
print_iter = 50
visual_iter = 500
eval_iter = 500
LAM = 10.
BALANCE = 1.

device = 'cuda' #'cpu'
name = 'cyclegan-test-L1'

A = 'jj'
B = 'eason'


singers = ['jj', 'eason']

E = models.Encoder().to(device)

G_a = models.Generator(A).to(device)
G_b = models.Generator(B).to(device)

D_a = models.Discriminator(A).to(device)
D_b = models.Discriminator(B).to(device)

loader_a = torch.utils.data.DataLoader(dataset.SongSegs(singer=A, len=batch_size * total_iter), batch_size=batch_size, shuffle=True, num_workers=10)
loader_b = torch.utils.data.DataLoader(dataset.SongSegs(singer=B, len=batch_size * total_iter), batch_size=batch_size, shuffle=True, num_workers=10)

#iterate manually
it_a = iter(loader_a)
it_b = iter(loader_b)

def calc_cycle_loss(batch_x, rec_x):
    return f.l1_loss(batch_x, rec_x) / batch_x.size(0) / batch_x.size(1)
    # return f.mse_loss(batch_x, rec_x) / batch_x.size(0) / batch_x.size(1)

#maximize delta
def calc_gan_loss(delta):
    return -torch.log(torch.sigmoid(delta)).mean()

for ix in range(total_iter):
    curr_lr = adjust_lr(init_lr, ix, total_iter)
    G_optim = optim.Adam(list(E.parameters()) + list(G_a.parameters()) + list(G_b.parameters()), lr=curr_lr, betas=(0.5, 0.999))
    D_optim = optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr=curr_lr, betas=(0.5, 0.999))

    batch_a = next(it_a).to(device)
    batch_b = next(it_b).to(device)

    fake_b = G_a(E(batch_a))
    fake_a = G_b(E(batch_b))

    rec_a = G_b(E(fake_b))
    rec_b = G_a(E(fake_a))

    pred_fake_a = D_a(fake_a)
    pred_fake_b = D_b(fake_b)
    pred_real_a = D_a(batch_a)
    pred_real_b = D_b(batch_b)

    #calc G loss...
    gan_loss = calc_gan_loss(pred_fake_a - pred_real_a) + calc_gan_loss(pred_fake_b - pred_real_b)
    cycle_loss = LAM * (calc_cycle_loss(batch_a, rec_a) + calc_cycle_loss(batch_b, rec_b))
    G_loss = gan_loss + cycle_loss

    #calc D loss...
    D_loss = calc_gan_loss(pred_real_a - pred_fake_a) + calc_gan_loss(pred_real_b - pred_fake_b)

    loss = G_loss + BALANCE * D_loss

    # print('G: gan loss {}, cycle_loss {}'.format(float(gan_loss), float(cycle_loss)))
    # print('D: gan loss {}'.format(float(D_loss)))

    D_optim.zero_grad()
    G_optim.zero_grad()

    loss.backward()

    D_optim.step()
    G_optim.step()

    if ix % 10 == 0:
        print('ix: [{}/{}] G: gan {:.4f}, cycle {:.4f}, total {:.4f}, D: total {:.4f}'.format(ix, total_iter, float(gan_loss), float(cycle_loss), float(G_loss), float(D_loss)))

    if ix % eval_iter == 0:
        eval.eval_file('./raw_data/jj/林俊杰 - 江南.flac', 'results/jj-to-eason-{}-{}.wav'.format(name, ix), encoder=E, geneator=G_a)

