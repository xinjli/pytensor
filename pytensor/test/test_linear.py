from pytensor.data.digit_dataset import *
from pytensor.test.common import *
from pytensor.model.linear import *


class Linear(Graph):

    def __init__(self, input_size, output_size):
        super().__init__("Linear")

        # make graph
        self.affine = self.get_operation('Affine', {'input_size' : input_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_tensor):
        affine_tensor = self.affine.forward(input_tensor)
        return self.softmaxloss.forward(affine_tensor)

    def loss(self, target_tensor):
        return self.softmaxloss.loss(target_tensor)


class TestLinearModel(unittest.TestCase):

    def test_gradient(self):

        """
        validate model's gradient with numerical methods

        :return:
        """

        data_train, data_test, label_train, label_test = digit_dataset()
        model = Linear(64,10)

        grad_info = gradient_generator(model, Tensor([data_train[0]]), Tensor([label_train[0]]))

        for var, expected_grad, actual_grad in grad_info:
            diff = np.sum(np.abs(expected_grad - actual_grad))
            print("Now checking: ", var)
            self.assertLessEqual(diff, 0.001)

if __name__ == '__main__':
    unittest.main()