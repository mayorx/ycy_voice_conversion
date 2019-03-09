import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np



#input && output of AutoEncoder

def input_output_vis(input, output, file_name):
    SAMPLE_VIS_DIR = './debug/input_and_output'
    assert len(input.shape) == 1 and input.shape[0] == output.shape[0]

    fig, ax = plt.subplots()
    t = np.linspace(0., 1., input.shape[0])
    ax.plot(t, input, t, output)
    plt.savefig(os.path.join(SAMPLE_VIS_DIR, file_name))
    plt.clf()
