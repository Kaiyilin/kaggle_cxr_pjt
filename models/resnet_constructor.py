import sys
import tensorflow as tf 

class ResNetConstructor():
    def __init__(self, input_shape) -> None:
        self.input_shape = input_shape
    
    def build_resnet_ssl(self, **kwargs):
        """
        model: str, ssl(self supervised) or supv(supervised), case insensitive
        """
        model = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            input_shape=self.input_shape,
            weights=None,
            **kwargs
        )
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = tf.keras.Model(inputs=model.input, outputs=x)
        return model

    def build_resnet_supv(self, hidden_dense_unit=1024, num_classes=4, **kwargs):
        base_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=self.input_shape,
            weights=None
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(hidden_dense_unit, activation='relu')(x)
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        return model 

        
"""
class ResNetConstructor():
    def build_resnet(**kwargs):
        model = tf.keras.applications.resnet.ResNet50(
            **kwargs
        )
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = tf.keras.Model(inputs=model.input, outputs=x)
        return model
"""