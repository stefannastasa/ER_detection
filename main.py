# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from data_splitting.dataSet import dataSet
import tensorflow as tf
from intermediate.models.base_model.classification_model import GoogLeNet
import matplotlib.pyplot as plt
import numpy as np
def gpuConfig():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(gpus[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=5126)])
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = dataSet(60)


    gpuConfig()

    model = GoogLeNet().getModel()

    model.compile(optimizer='adam',loss=tf.losses.BinaryCrossentropy(from_logits=1), metrics=['accuracy'])
    input_train = np.array([p[0] for p in data.trainingSet]).reshape(-1, 400, 280, 1)
    output_train = np.asarray([p[1] for p in data.trainingSet])

    input_test = np.array([p[0] for p in data.validationSet]).reshape(-1, 400, 280, 1)
    output_test = np.asarray([p[1] for p in data.validationSet])

    history = model.fit(input_train, output_train, batch_size=len(data.trainingSet)//14, epochs=100 ,validation_data=(input_test, output_test))
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test', 'train'], loc='upper left')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
