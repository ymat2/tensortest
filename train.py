from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from dataset import load_mnist_dataset


def build_sequential_model(data_shape):
    n_input_layer = 256
    n_hidden_layer = 128
    drop_rate = 0.5
    n_output_layer = 10

    model = Sequential()
    model.add(Dense(n_input_layer, activation = 'sigmoid', input_shape = (data_shape,)))  # input layer
    model.add(Dense(n_hidden_layer, activation = 'sigmoid'))  # hidden layer
    model.add(Dropout(rate = drop_rate))                      # drop out
    model.add(Dense(n_output_layer, activation = 'softmax'))  # output layer

    return model


def save_history_as_fig(history):
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['acc'], label = 'Training accuracy')
    plt.plot(history.history['val_acc'], label = 'Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.savefig('./accuracy.png')


def train(n_epochs, model_path, save_history = True, evaluate = True):
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset()
    data_shape = x_train.shape[1]

    model = build_sequential_model(data_shape = data_shape)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = SGD(lr = 0.1),
        metrics=['acc']
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size = 500,
        epochs = n_epochs,
        validation_split = 0.2
    )

    if model_path:
        model.save(model_path)

    if save_history:
        save_history_as_fig(history)

    if evaluate:
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('##### EVALUATE #####')
        print('loss: {:.3f}\naccuracy: {:.3f}'.format(test_loss, test_acc))
        print('####################')


def main():
    n_epochs = 5
    model_path = "models/full_mnist_model.h5"
    train(
        n_epochs = n_epochs,
        model_path = model_path,
        save_history = True,
        evaluate = True
    )


if __name__ == "__main__":
    main()
