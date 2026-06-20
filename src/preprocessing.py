import tensorflow as tf
from tensorflow import keras
import keras.saving as keras_saving

@keras_saving.register_keras_serializable(package="preprocessing")
def preprocess_efficientnetv2b3(x):
    return tf.keras.applications.efficientnet_v2.preprocess_input(x)

@keras_saving.register_keras_serializable(package="preprocessing")
def preprocess_resnet50v2(x):
    return tf.keras.applications.resnet_v2.preprocess_input(x)

@keras_saving.register_keras_serializable(package="preprocessing")
def preprocess_mobilenetv3large(x):
    return tf.keras.applications.mobilenet_v3.preprocess_input(x)

@keras_saving.register_keras_serializable(package="preprocessing")
def preprocess_convnexttiny(x):
    return tf.keras.applications.convnext.preprocess_input(x)

ALL_CUSTOM_OBJECTS = {
    "preprocess_efficientnetv2b3": preprocess_efficientnetv2b3,
    "preprocess_resnet50v2":       preprocess_resnet50v2,
    "preprocess_mobilenetv3large": preprocess_mobilenetv3large,
    "preprocess_convnexttiny":     preprocess_convnexttiny,
}
