import torch
import torch.utils.data
import models_conv as models
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import eval
from utils import *

batch_size = 64
init_lr = 1e-5
total_iter = 300001
ckpt_iter = 1000
print_iter = 50
visual_iter = 500
eval_iter = 2000
LAM = 10.
BALANCE = 1.

device = 'cuda' #'cpu'
name = 'cyclegan-with-strongD'

A = 'jj'
B = 'jay'


# singers = ['jj', 'eason']

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
    # return f.l1_loss(batch_x, rec_x) / batch_x.size(0) / batch_x.size(1)
    # print('batch x , rec x', batch_x.size(), rec_x.size())
    return f.mse_loss(batch_x, rec_x) / batch_x.size(0) / batch_x.size(1)

#maximize delta
# def calc_gan_loss(delta):
#     return -torch.log(torch.sigmoid(delta)).mean()

def calc_gan_loss(pred, is_one=True):
    if is_one:
        target = torch.ones(pred.size()).to(device)
    else:
        target = torch.zeros(pred.size()).to(device)
    return f.mse_loss(pred, target, size_average=True)
    # return f.binary_cross_entropy_with_logits(pred, target, size_average=True)

#borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py#L214-L225
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

cnt_G = 0
cnt_D = 0

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

    #Generator
    set_requires_grad([D_a, D_b], requires_grad=False)

    gan_loss = calc_gan_loss(D_a(fake_a), True) + calc_gan_loss(D_b(fake_b), True)
    identity_loss =  0.5 * (calc_cycle_loss(batch_b, G_a(E(batch_b))) + calc_cycle_loss(batch_a, G_b(E(batch_a))))
    cycle_loss = LAM * (calc_cycle_loss(batch_a, rec_a) + calc_cycle_loss(batch_b, rec_b))

    G_loss = gan_loss + cycle_loss + identity_loss
    G_optim.zero_grad()
    G_loss.backward()
    G_optim.step()


    # batch_b == G_a(E(batch_b))
    # batch_a == G_b(E(batch_a))


    #calc G loss...
    # gan_loss = calc_gan_loss(pred_fake_a - pred_real_a) + calc_gan_loss(pred_fake_b - pred_real_b)
    # gan_loss = calc_gan_loss(pred_fake_a, True) + calc_gan_loss(pred_fake_b, True)
    # identity_loss = 0.5 * (calc_cycle_loss(batch_b, G_a(E(batch_b))) + calc_cycle_loss(batch_a, G_b(E(batch_a))))

    # cycle_loss = LAM * (calc_cycle_loss(batch_a, rec_a) + calc_cycle_loss(batch_b, rec_b))
    # G_loss = gan_loss + cycle_loss + identity_loss


    #Discriminator
    set_requires_grad([D_a, D_b], requires_grad=True)

    pred_fake_a = D_a(fake_a.detach())
    pred_fake_b = D_b(fake_b.detach())
    pred_real_a = D_a(batch_a)
    pred_real_b = D_b(batch_b)

    #calc D loss...
    D_a_loss = calc_gan_loss(pred_real_a, True) + calc_gan_loss(pred_fake_a, False)
    D_b_loss = calc_gan_loss(pred_real_b, True) + calc_gan_loss(pred_fake_b, False)
    D_loss = 0.5 * (D_a_loss + D_b_loss)

    D_optim.zero_grad()
    D_loss.backward()
    D_optim.step()

    # D_loss = calc_gan_loss(pred_real_a, True) + calc_gan_loss(pred_fake_b, 0)
    # D_loss = calc_gan_loss(pred_real_a - pred_fake_a) + calc_gan_loss(pred_real_b - pred_fake_b)

    # if (gan_loss > D_loss and np.random.rand() < 0.9) or ():
    #     loss = gan_loss + cycle_loss
    #     cnt_G += 1
    # else:
    #     loss = D_loss
    #     cnt_D += 1
    #     if ix % 2 == 0:
    #         loss += cycle_loss

    # if ix % 2 == 0:
    #     loss = G_loss
    # else:
    #     loss = D_loss

    # print('G: gan loss {}, cycle_loss {}'.format(float(gan_loss), float(cycle_loss)))
    # print('D: gan loss {}'.format(float(D_loss)))

    # D_optim.zero_grad()
    # G_optim.zero_grad()

    # loss.backward()

    # D_optim.step()
    # G_optim.step()

    if ix % 10 == 0:
        print('ix: [{}/{}]'
              '\n\tG: gan {:.4f}, cycle {:.4f}, idt {:.4f} total {:.4f}, cnt {}'
              '\n\tD: D_a {:.4f}, D_b {:.4f} total {:.4f}, cnt {}'
              .format(ix, total_iter,
                      float(gan_loss), float(cycle_loss), float(identity_loss), float(G_loss), cnt_G,
                      float(D_a_loss), float(D_b_loss), float(D_loss), cnt_D)
              )

    if (ix+1) % eval_iter == 0:
        eval.eval_file('./raw_data/jj/林俊杰 - 江南.flac', 'results/jj-to-eason-{}-{}.wav'.format(name, ix), encoder=E, geneator=G_a)

