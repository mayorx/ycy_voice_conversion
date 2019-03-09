import unittest
import utils
import models_conv as models

class YCY_TEST(unittest.TestCase):

    def test_chekpoint(self):
        encoder = models.Encoder().to('cuda')
        generators = {}

        singers = ['jj']
        for singer in singers:
            generators[singer] = models.Generator(singer).to('cuda')

        utils.save_checkpoint(encoder, generators, 'debug')
        utils.load_checkpoint(encoder, generators, 'debug')


if __name__ == '__main__':
    unittest.main()