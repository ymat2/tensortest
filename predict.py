from tensorflow.keras.models import load_model
from PIL import Image

import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    args = parser.parse_args()
    return args


def read_image_as_array(file):
    img = Image.open(file)
    img = img.convert("L").resize((28, 28))
    img = np.asarray(img)
    return img

def pred(file, model_path):
    img = read_image_as_array(file)
    img = img.reshape((1, 784))
    model = load_model(model_path)

    test_predictions = model.predict(img)
    test_predictions = np.argmax(test_predictions, axis = 1)
    return test_predictions


def main():
    args = parse_arguments()
    file = args.input
    model = "models/full_mnist_model.h10"
    result = pred(file, model_path = model)
    print(result)


if __name__ == "__main__":
    main()
