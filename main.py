import os, datetime, argparse, logging
import tensorflow as tf
from tensorflow.python.eager.monitoring import Metric
from dataloader.dataloader import (
    train_datagenator, 
    val_datagenator, 
    test_datagenator
    )
from configs import pjt_configs
from models.resnet_constructor import ResNetConstructor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default=pjt_configs["records"]["log_path"])
    parser.add_argument('--ckpt_path', type=str, default=pjt_configs["records"]["ckpt_path"])
    parser.add_argument('--input_shape', default=pjt_configs["training"]["shape"])
    parser.add_argument('--batchsize', type=int, default=pjt_configs["training"]["batch_size"])
    parser.add_argument('--epochs', type=int, default=pjt_configs["training"]["epochs"])
    
    #logging.basicConfig(filename="Info.log", level=logging.INFO, format=)
    
    # Parse the argument and store it in a dictionary:
    args = vars(parser.parse_args())
    excutime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = args["log_path"] + excutime + "/"
    ckpt_dir = args["ckpt_path"] + excutime + "/"
    
    try:
        os.makedirs(tb_log_dir)
        os.makedirs(ckpt_dir)
    except FileExistsError as e:
        print(e)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPU in use of Mirrored strategy: {strategy.num_replicas_in_sync}")
    
    # remeber to rewrite the model, it's completely messy
    with strategy.scope():
        model = ResNetConstructor(input_shape=args["input_shape"]).build_resnet_supv()

        tb_callbacks = [ 
                    tf.keras.callbacks.TensorBoard(
                        log_dir=tb_log_dir, 
                        histogram_freq=1, 
                        write_graph=True, 
                        write_images=False,
                        update_freq='epoch', 
                        profile_batch=2, 
                        embeddings_freq=0,
                        embeddings_metadata=None
                        )]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.AUC()
            ]
        )

        train_datagenator.batch_size = args["batchsize"]

        history = model.fit(
            train_datagenator,
            validation_data=val_datagenator,
            epochs=args["epochs"],
            callbacks=tb_callbacks
        )
