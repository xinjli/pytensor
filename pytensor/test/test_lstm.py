from pytensor.data.digit_dataset import *
from pytensor.test.common import *
from pytensor.model.lstm import *


class TestLSTMModel(unittest.TestCase):

    def test_gradient(self):
        """
        validate model's gradient with numerical methods

        :return:
        """

        input_lst = [np.random.randint(5) for i in range(10)]
        output_lst = [np.random.randint(5) for i in range(10)]

        model = LSTMLM(5, 5, 10)

        grad_info = gradient_generator(model, input_lst, output_lst)

        for var, expected_grad, actual_grad in grad_info:
            diff = np.sum(np.abs(expected_grad - actual_grad))
            print("Now checking ", var)
            self.assertLessEqual(diff, 0.001)

if __name__ == '__main__':
    unittest.main()