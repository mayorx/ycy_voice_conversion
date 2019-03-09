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

    def test_adjust_lr(self):
        self.assertEqual(utils.adjust_lr(1., 100, 100), 0)
        self.assertEqual(utils.adjust_lr(1., 0, 100), 1)
        self.assertAlmostEqual(utils.adjust_lr(1., 50, 100), 0.5 ** 0.9)


if __name__ == '__main__':
    unittest.main()