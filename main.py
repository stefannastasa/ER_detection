# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from data_splitting.dataSet import dataSet
import tensorflow as tf
from intermediate.models.base_model.classification_model import GoogLeNet
from intermediate.models.base_model.updated_model import ModifiedGoogLeNet
import matplotlib.pyplot as plt
import numpy as np


def gpuConfig():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(gpus[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=5000)])
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            print("Hello")
        except RuntimeError as e:
            print(e)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = dataSet(75)

    model_name = "with_filters"

    gpuConfig()

    input_train = np.array([p[0] for p in data.trainingSet]).reshape(-1, 287, 200, 3)
    output_train = np.asarray([p[1] for p in data.trainingSet])

    input_test = np.array([p[0] for p in data.validationSet]).reshape(-1, 287, 200, 3)
    output_test = np.asarray([p[1] for p in data.validationSet])

    if model_name == "googlenet":

        model = GoogLeNet().getModel()

        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

        history = model.fit(input_train, output_train, batch_size=len(data.trainingSet) // 11, epochs=25,
                            validation_data=(input_test, output_test))
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.show()
    elif model_name == "with_filters":
        model = ModifiedGoogLeNet().getModel()
        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

        history = model.fit(input_train, output_train, batch_size=len(data.trainingSet) // 11, epochs=25,
                            validation_data=(input_test, output_test))

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.show()
    else:
        base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(287, 200, 3))
        # base_model.trainable = tf.keras.applications.resnet_v2.preprocess_input(weights='imagenet', include_top=False, input_shape=(287, 200, 3))
        # base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(287, 200, 3))
        base_model.trainable = False

        feature_batch = base_model(input_train)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        prediction_layer = tf.keras.layers.Dense(1)

        model = tf.keras.models.Sequential([base_model, global_average_layer, prediction_layer])

        model.summary()

        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(from_logits=1), metrics=['accuracy'])

        history = model.fit(input_train, output_train, batch_size=len(data.trainingSet) // 5, epochs=40,
                            validation_data=(input_test, output_test))

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='upper left')
        plt.show()

        pass
