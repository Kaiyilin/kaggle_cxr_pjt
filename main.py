import os 
import tensorflow as tf
from tensorflow.python.eager.monitoring import Metric
from dataloader.dataloader import (
    train_datagenator, 
    val_datagenator, 
    test_datagenator
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_shape=(256, 256, 3),
        weights=None
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 4 classes
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
    # this is the model we will train
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )

    model.fit(
        train_datagenator,
        validation_data=val_datagenator,
        epochs=100
    )