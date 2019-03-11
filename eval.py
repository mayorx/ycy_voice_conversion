import librosa
import utils
import models_conv as models
import torch

SAMPLE_LENGTH_LIMIT = 700000

def generate_voice_conversion(samples, encoder, generator):
    input = torch.from_numpy(samples)
    input = input.unsqueeze(0).to('cuda')
    code = encoder(input)
    output = generator(code).squeeze().detach().cpu().numpy()
    return output


def eval_file(in_file, out_file, encoder, geneator, vis=True):
    print('eval .. {} to {}'.format(in_file, out_file))
    samples, _ = librosa.load(in_file, 20000)
    if len(samples) > 2 * SAMPLE_LENGTH_LIMIT:
        print('length limit .. {} seconds.'.format(SAMPLE_LENGTH_LIMIT / 20000))
        samples = samples[SAMPLE_LENGTH_LIMIT: SAMPLE_LENGTH_LIMIT + SAMPLE_LENGTH_LIMIT]
        print('.... cutted.')
    output = generate_voice_conversion(samples, encoder, geneator)
    librosa.output.write_wav(out_file, output, 20000)
    if vis:
        utils.input_output_vis(samples, output, 'eval')
    print('eval finished.')

if __name__ == '__main__':
    singers = ['jj']
    encoder = models.Encoder().to('cuda')
    generators = {}

    for singer in singers:
        generators[singer] = models.Generator(singer).to('cuda')

    utils.load_checkpoint(encoder, generators, 'ae-iter-0009000')

    eval_file('./data/burn.wav', './results/burn-jj.wav', encoder, generators['jj'])


