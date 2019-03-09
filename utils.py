import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np


SAMPLE_VIS_DIR = './debug/input_and_output'
CKPT_DIR = './ckpt'

#input && output of AutoEncoder

def input_output_vis(input, output, file_name):
    assert len(input.shape) == 1 and input.shape[0] == output.shape[0]

    fig, ax = plt.subplots()
    t = np.linspace(0., 1., input.shape[0])
    ax.plot(t, input, t, output)
    plt.savefig(os.path.join(SAMPLE_VIS_DIR, '{}.png'.format(file_name)))
    plt.clf()

def save_checkpoint(encoder, generators, ckpt_name):
    print('save checkpoint .... name:{}'.format(ckpt_name))
    torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, 'encoder-{}.pth'.format(ckpt_name)))
    for key in generators:
        generator = generators[key]
        torch.save(generator.state_dict(), os.path.join(CKPT_DIR, 'generator-{}-{}.pth'.format(key, ckpt_name)))
    print('checkpoint saved...')

def load_checkpoint(encoder, generators, ckpt_name):
    print('load checkpoint .... name:{}'.format(ckpt_name))
    encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'encoder-{}.pth'.format(ckpt_name))))
    for key in generators:
        generators[key].load_state_dict(torch.load(os.path.join(CKPT_DIR, 'generator-{}-{}.pth'.format(key, ckpt_name)))
    print('checkpoint loaded...')
    return encoder, generators

