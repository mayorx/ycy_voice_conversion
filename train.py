import torch
import torch.utils.data
import models_conv as models
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from utils import *

batch_size = 512
init_lr = 1e-4
total_iter = 100001
ckpt_iter = 1000
print_iter = 50
visual_iter = 500
name = 'x-ae'

# singers = ['dt', 'jj', 'eason']
singers = ['jj']

encoder = models.Encoder().to('cuda')
# eoptim = optim.SGD(encoder.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
# eoptim = optim.Adam(encoder.parameters(), lr=init_lr, betas=(0.5, 0.999))

generators = {}
loaders = {}
# goptims = {}

for singer in singers:
    generators[singer] = models.Generator(singer).to('cuda')
    ds = dataset.SongSegs(singer=singer, len=batch_size * total_iter)
    loaders[singer] = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)
    # goptims[singer] = optim.SGD(generators[singer].parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
    # goptims[singer] = optim.Adam(generators[singer].parameters(), lr=init_lr, betas=(0.5, 0.999))

for singer in singers:
    loader = loaders[singer]
    generator = generators[singer]

    total_loss = 0
    generator.train()
    encoder.train()

    for ix, input in enumerate(loader):
        curr_lr = adjust_lr(init_lr, ix, total_iter)
        eoptim = optim.Adam(encoder.parameters(), lr=curr_lr, betas=(0.5, 0.999))
        goptim = optim.Adam(generator.parameters(), lr=curr_lr, betas=(0.5, 0.999))

        input = input.to('cuda')
        code = encoder(input)

        output = generators[singer](code)
        #want input == output
        # print('input size {} , output size {}'.format(input.size(), output.size()))
        loss = f.mse_loss(input, output) / batch_size / 1000
        # print('ix: {}, loss: {}'.format(ix, float(loss)))
        total_loss += float(loss)

        # print((input - output).shape)
        # print(((input - output) ** 2).shape)

        goptim.zero_grad()
        eoptim.zero_grad()
        loss.backward()
        eoptim.step()
        goptim.step()

        if (ix+1) % print_iter == 0:
            print('ix: {}/{}, lr: {}, avg loss: {}'.format(ix, total_iter, curr_lr, total_loss / 10), flush=True)
            total_loss =  0.

        if ix % visual_iter == 0:
            print('input && output ... ')
            print(float(input.mean()), float(input.min()), float(input.max()), flush=True)
            print(float(output.mean()), float(output.min()), float(output.max()), flush=True)
            # input_output_vis(input[0].detach().squeeze().cpu().numpy(), output[0].detach().squeeze().cpu().numpy(), 'epooch-{}-iter-{}.png'.format(epoch, ix))
            input_output_vis(input[0].detach().squeeze().cpu().numpy(), output[0].detach().squeeze().cpu().numpy(), '{}-iter-{:07}'.format(name, ix))

        if ix % ckpt_iter == 0:
            generators[singer] = generator
            save_checkpoint(encoder, generators, '{}-iter-{:07}'.format(name, ix))



