from pytensor import *
from pytensor.data.digit_dataset import *
from pytensor.test.common import *

class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        self.graph = Graph("MLP")

        # make graph
        self.affine1 = self.graph.get_operation('Affine', {'input_size': input_size, 'hidden_size': hidden_size})
        self.sigmoid = self.graph.get_operation('Sigmoid')
        self.affine2 = self.graph.get_operation('Affine', {'input_size': hidden_size, 'hidden_size': output_size})
        self.softmaxloss = self.graph.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine1_variable = self.affine1.forward(input_variable)
        sigmoid_variable = self.sigmoid.forward(affine1_variable)
        affine2_variable = self.affine2.forward(sigmoid_variable)

        return self.softmaxloss.forward(affine2_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)



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