class Model:

    def forward(self, input_variables):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def loss(self, target):
        raise NotImplementedError