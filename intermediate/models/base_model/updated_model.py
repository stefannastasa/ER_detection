import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from scipy.ndimage import binary_dilation, label
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Sequential


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if sigma == 0:
        sigma = 1.0
    coords = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image: tf.Tensor, kernel_size: int, sigma: float) -> tf.Tensor:
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    kernel = np.reshape(kernel, (*kernel.shape, 1, 1))
    kernel = np.tile(kernel, (1, 1, image.shape[-1], 1))

    return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")


def pleura_preprocessing_np(image_batch):
    # Gaussian blur
    blurred_images = apply_gaussian_blur(image_batch, kernel_size=5, sigma=0)

    # Resize the images
    resized_images = tf.image.resize_with_crop_or_pad(blurred_images, 287, 200)

    # Sobel filter pe y
    sobel_y = tf.image.sobel_edges(resized_images)[..., 1]

    # Threshold on sobel
    sobel_threshold = np.abs(sobel_y.numpy()) > 30

    # Threshold on intensity
    intensity_threshold = resized_images.numpy() > 200

    combined_threshold = tf.convert_to_tensor(sobel_threshold * intensity_threshold, dtype=tf.float32)

    # Dilation
    dilated_images = binary_dilation(combined_threshold, iterations=1)

    components, num_components = label(dilated_images)

    # Keep the largest area region
    largest_components = np.float32(components == np.argmax(np.bincount(components.flat)[1:]) + 1)
    resized_output = tf.expand_dims(largest_components, axis=-1)[:, :, :, 0]

    return resized_output


# Modified Inception model
class ModifiedInception(tf.keras.models.Model):
    def __init__(self, c1, c2, c3, c4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.b1_1 = Conv2D(c1, kernel_size=1)
        self.b2_1 = Conv2D(c2[0], kernel_size=1)
        self.b2_2 = Conv2D(c2[1], kernel_size=3, padding="same")
        self.b3_1 = Conv2D(c3[0], kernel_size=1)
        self.b3_2 = Conv2D(c3[1], kernel_size=5, padding="same")
        self.b4_1 = MaxPooling2D(3, padding="same", strides=1)
        self.b4_2 = Conv2D(c4, kernel_size=1)

    def call(self, inputs, **kwargs):
        b1 = relu(self.b1_1(inputs))
        b2 = relu(self.b2_2(relu(self.b2_1(inputs))))
        b3 = relu(self.b3_2(relu(self.b3_1(inputs))))
        b4 = relu(self.b4_2(self.b4_1(inputs)))

        return tf.concat([b1, b2, b3, b4], 3)

# Modified GoogLeNet model
class ModifiedGoogLeNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.net = self.build_model()

    def getModel(self):
        return self.net

    def build_model(self):
        input_layer = layers.Input(shape=(287, 200, 1))
        preprocessed_input = layers.Lambda(lambda x: tf.numpy_function(pleura_preprocessing_np, [x], tf.float32))(input_layer)

        x = self.b1()(preprocessed_input)
        x = self.b2()(x)
        x = self.b3()(x)
        x = self.b4()(x)
        x = self.b5()(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def b1(self):
        return tf.keras.Sequential([Conv2D(64, kernel_size=7, strides=2, padding="same", activation="relu", input_shape=(287, 200, 1)),
                                    MaxPooling2D(3, strides=2, padding="same")])

    def b2(self):
        return tf.keras.Sequential([Conv2D(64, kernel_size=1, activation="relu"),
                                    Conv2D(192, kernel_size=3, activation="relu"),
                                    MaxPooling2D(3, strides=2, padding="same")])

    def b3(self):
        return tf.keras.Sequential([
            ModifiedInception(64, (96, 128), (16, 32), 32),
            ModifiedInception(128, (128, 192), (32, 96), 64),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(3, strides=2, padding="same")
        ])

    def b4(self):
        return tf.keras.Sequential([
            ModifiedInception(192, (96, 208), (16, 48), 64),
            ModifiedInception(160, (112, 224), (24, 64), 64),
            ModifiedInception(128, (128, 256), (24, 64), 64),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(3, strides=2, padding="same")
        ])

    def b5(self):
        return tf.keras.Sequential([
            ModifiedInception(256, (160, 320), (32, 128), 128),
            ModifiedInception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAveragePooling2D(),  # Using GlobalAveragePooling2D instead of AveragePooling2D
            Dense(1, activation='sigmoid')
        ])