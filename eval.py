import librosa
import utils
import models_conv as models
import torch

def generate_voice_conversion(samples, encoder, generator):
    input = torch.from_numpy(samples)
    print('input size', input.size())
    input = input.unsqueeze(0).to('cuda')
    code = encoder(input)
    output = generator(code).squeeze().detach().cpu().numpy()
    return output


def eval_file(in_file, out_file, encoder, geneator, vis=True):
    samples, _ = librosa.load(in_file, 20000)
    output = generate_voice_conversion(samples, encoder, geneator)
    librosa.output.write_wav(out_file, output, 20000)
    if vis:
        utils.input_output_vis(samples, output, 'eval')

if __name__ == '__main__':
    singers = ['jj']
    encoder = models.Encoder().to('cuda')
    generators = {}

    for singer in singers:
        generators[singer] = models.Generator(singer).to('cuda')

    utils.load_checkpoint(encoder, generators, 'ae-iter-0009000')

    eval_file('./data/burn.wav', './results/burn-jj.wav', encoder, generators['jj'])


