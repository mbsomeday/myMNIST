from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2


def get_mnist_data(mnist_path):
    with np.load(mnist_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(images, labels, num_classes):
    images = images.astype("float32") / 255
    images = np.expand_dims(images, -1)
    
    labels = keras.utils.to_categorical(labels, num_classes)
    
    return images, labels

# ok
def get_model(num_classes):
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


    



if __name__ == '__main__':
    num_classes = 10
    input_shape = (28, 28, 1)
    batch_size = 128
    mnist_path = r'./mnist.npz'

    (x_train, y_train), (x_test, y_test) = get_mnist_data(mnist_path)
    x_train, y_train = preprocess_data(x_train, y_train, num_classes)
    x_test, y_test = preprocess_data(x_test, y_test, num_classes)
    # model = get_model(num_classes)
    #
    # model.compile(loss="categorical_crossentropy",
    #               optimizer="adam",
    #               metrics=["accuracy"])
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_split=0.1)
    # model.save(r'./model/mnist_model.h5')
    
    model = get_model(num_classes)
    model_path = r'./model/mnist_model.h5'
    model(np.ones(shape=(1, 28, 28, 1), dtype="float32"))
    model.load_weights(model_path)
    # model.compile(loss="categorical_crossentropy",
    #               optimizer="adam",
    #               metrics=["accuracy"])
    res = model.predict(x_test)
    r = np.argmax(res, axis=1)
    






















