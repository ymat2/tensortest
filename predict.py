from tensorflow.keras.models import load_model
from dataset import load_mnist_dataset
import matplotlib.pyplot as plt
import numpy as np


def pred(model_path):
    model = load_model(model_path)
    (_, _), (x_test, _) = load_mnist_dataset()

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[i].reshape((28, 28)), 'gray')
    plt.show()

    test_predictions = model.predict(x_test[0:5])
    test_predictions = np.argmax(test_predictions, axis = 1)
    print(test_predictions)


def main():
    pred(model_path = "models/full_mnist_model.h5")


if __name__ == "__main__":
    main()
