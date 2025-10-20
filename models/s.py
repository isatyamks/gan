import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow:", tf.__version__)
print("TFDS:", tfds.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
