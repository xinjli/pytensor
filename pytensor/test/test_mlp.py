from pytensor.data.digit_dataset import *
from pytensor.test.common import *
from pytensor.model.mlp import *


class TestMLPModel(unittest.TestCase):

    def test_gradient(self):
        """
        validate model's gradient with numerical methods

        :return:
        """

        data_train, data_test, label_train, label_test = digit_dataset()
        model = MLP(64, 30, 10)

        grad_info = gradient_generator(model, Variable([data_train[0]]), Variable([label_train[0]]))

        for var, expected_grad, actual_grad in grad_info:
            diff = np.sum(np.abs(expected_grad - actual_grad))
            print("Now checking ", var)
            self.assertLessEqual(diff, 0.001)

if __name__ == '__main__':
    unittest.main()