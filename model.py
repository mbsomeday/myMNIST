from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import cfg


def CNN(cfg):
    inputs = keras.Input(shape=(28, 28, 1))
    c1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1")(inputs)
    p1 = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool1')(c1)
    c2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2")(p1)
    p2 = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool2')(c2)
    flat = layers.Flatten(name="flatten")(p2)
    drop = layers.Dropout(0.5)(flat)
    outputs = layers.Dense(cfg.num_classes, activation="relu")(drop)
    
    model = keras.Model(inputs, outputs, name='mnist model')
    
    return model



if __name__ == '__main__':
    model = CNN(cfg)
    # model.summary()





















