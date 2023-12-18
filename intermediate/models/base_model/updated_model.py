import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.activations import relu
import tensorflow.keras.layers as layers
from scipy.ndimage import binary_dilation, label

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generates a Gaussian kernel."""
    if sigma == 0:
        sigma = 1.0
    coords = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image: tf.Tensor, kernel_size: int, sigma: float) -> tf.Tensor:
    """Applies Gaussian blur to an image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    kernel = np.reshape(kernel, (*kernel.shape, 1, 1))
    kernel = np.tile(kernel, (1, 1, image.shape[-1], 1))

    return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")


def pleura_preprocessing_np(image):
    image_np = np.array(image)

    #gaussian blur
    blurred_image = apply_gaussian_blur(image_np, kernel_size=5, sigma=0)

    resized_image = cv2.resize(blurred_image, (150, 150,1))

    # Sobel filter pe y
    sobel_y = cv2.Sobel(resized_image, cv2.CV_64F, 0, 1, ksize=3)

    # Threshold on sobel
    sobel_threshold = np.float32(np.abs(sobel_y) > 30)

    # Threshold on intensity
    intensity_threshold = np.float32(resized_image > 200)

    combined_threshold = tf.convert_to_tensor(sobel_threshold * intensity_threshold, dtype=tf.float32)

    # Dilation
    dilated_image = binary_dilation(combined_threshold, iterations=1)

    components, num_components = label(dilated_image)

    # Keep the largest area region
    largest_component = np.float32(components == np.argmax(np.bincount(components.flat)[1:]) + 1)

    # Reshape the output tensor to have a fixed shape
    return tf.reshape(tf.convert_to_tensor(largest_component), (287, 200, 1))

# Function to enhance pleura line visibility through contrast adjustment
def enhance_pleura_line(image):
    preprocessed_image = pleura_preprocessing(image)

    # Stack the preprocessed image to create a 3-channel image
    preprocessed_image = tf.stack([preprocessed_image, preprocessed_image, preprocessed_image], axis=-1)
    resized_image = tf.image.resize(preprocessed_image, (287, 200, 1))
    return resized_image



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
    def __init__(self, input_shape):
        super().__init__()
        self.inputShape=input_shape
        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(287,200,1)),
            tf.keras.layers.Lambda(lambda x: tf.numpy_function(pleura_preprocessing_np, [x], tf.float32)),
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(1, activation='sigmoid'),

        ])

    def getModel(self):
        return self.net

    def b1(self):
        return Sequential([Conv2D(64, kernel_size=7, strides=2, padding="same", activation="relu"),
                           MaxPooling2D(3, strides=2, padding="same")])

    def b2(self):
        return Sequential([Conv2D(64, kernel_size=1, activation="relu"),
                           Conv2D(192, kernel_size=3, activation="relu"),
                           MaxPooling2D(3, strides=2, padding="same")])

    def b3(self):
        return Sequential([
            ModifiedInception(64, (96, 128), (16, 32), 32),
            ModifiedInception(128, (128, 192), (32, 96), 64),
            # Additional convolutional layers for pleura line and Merlin's space
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(3, strides=2, padding="same")
        ])

    def b4(self):
        return Sequential([
            ModifiedInception(192, (96, 208), (16, 48), 64),
            ModifiedInception(160, (112, 224), (24, 64), 64),
            ModifiedInception(128, (128, 256), (24, 64), 64),
            # Additional convolutional layers for pleura line and Merlin's space
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(3, strides=2, padding="same")
        ])

    def b5(self):
        return Sequential([ModifiedInception(256, (160, 320), (32, 128), 128),
                           ModifiedInception(384, (192, 384), (48, 128), 128),
                           layers.AveragePooling2D(1),
                           Flatten()])

