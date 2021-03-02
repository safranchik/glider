from glider.losses import expected_cross_entropy
import torch
import random
import unittest


class TestWeakLosses(unittest.TestCase):

    def test_cross_entropy(self):
        num_runs = 1000

        strong_loss = torch.nn.functional.cross_entropy
        weak_loss = expected_cross_entropy

        num_classes = random.randint(1, 50)
        output_size = random.randint(1, 50)

        for i in range(num_runs):
            output = torch.rand(output_size, num_classes, dtype=torch.float32)
            Y = torch.randint(num_classes, (output_size,))

            Y_bar = torch.zeros(output_size, num_classes)
            Y_bar[range(output_size), Y] = 1

            assert torch.isclose(strong_loss(output, Y, reduction='mean'), weak_loss(output, Y_bar, reduction='mean'))
            assert torch.isclose(strong_loss(output, Y, reduction='sum'), weak_loss(output, Y_bar, reduction='sum'))
            assert torch.isclose(strong_loss(output, Y, reduction='none'),
                                 weak_loss(output, Y_bar, reduction='none')).all()


if __name__ == '__main__':
    unittest.main()
