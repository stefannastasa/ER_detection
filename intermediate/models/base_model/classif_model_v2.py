

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.activations import relu

class MobileNetTL():
    def __init__(self):
        self.net = models.Sequential([layers.Input((287, 200,1)), self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                                     layers.Dense(1)])

    def getModel(self):
        return self.net

    def b1(self):
        return models.Sequential([layers.Conv2D(64, kernel_size=7, strides=2, padding="same", activation="relu"),
                                 layers.MaxPool2D(3, strides=2, padding="same")])

    def b2(self):
        return models.Sequential(
            [layers.Conv2D(64, kernel_size=1, activation="relu"),
            layers.Conv2D(192, kernel_size=3, activation="relu"),
            layers.MaxPool2D(3, strides=2, padding="same")]
        )

    def b3(self):
        return models.Sequential(
            [Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            layers.MaxPool2D(3, strides=2, padding="same")]
        )

    def b4(self):
        return models.Sequential(
            [Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            layers.MaxPool2D(3, strides=2, padding="same")]
        )

    def b5(self):
        return models.Sequential(
            [Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            layers.AveragePooling2D(1),
            layers.Flatten()]
        )