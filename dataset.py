from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import argparse
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", default = False, type = bool)
    args = parser.parse_args()
    return args


def load_mnist_dataset(prnt = False):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    if prnt:
        print(train_labels[0:5])
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(train_images[i], 'gray')
        plt.show()

    train_images = train_images.reshape((train_images.shape[0], 784))
    test_images = test_images.reshape((test_images.shape[0], 784))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def main():
    args = parse_arguments()
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset(args.print)
    print("train_data:", x_train.shape)
    print("train_label:", y_train.shape)
    print("test_data:", x_test.shape)
    print("test_label:", y_test.shape)


if __name__ == "__main__":
    main()
