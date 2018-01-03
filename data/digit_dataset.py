from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def digit_dataset():

    digits = load_digits()
    digits.data /= 16.0
    data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

    return data_train, data_test, label_train, label_test
