import logging
import tensorflow as tf
import math
from decimal import Decimal
import numpy as np
 
class SimCLR(tf.keras.Model):

    def __init__(self, 
                 base_model,
                 projection_head) -> None:
        """The Network architecture and weights should be the same
        """
        super(SimCLR, self).__init__()
        self.base_model = base_model
        self.projection_head = projection_head

    def compile(self, optNet, loss_fn):
        super(SimCLR, self).compile()
        self.optNet = optNet
        self.loss = loss_fn

    def __str__():
        print("SimCLR is working")

    def train_step(self, data):
        """ Same backbone, but do twice before backpropagation"""
        img1 = data[0]
        img2 = data[1]

        with tf.GradientTape(persistent=True) as tape:
            representation1 = self.base_model(img1)
            representation2 = self.base_model(img2)
            representation1 = tf.math.l2_normalize(representation1, axis=1)
            representation2 = tf.math.l2_normalize(representation2, axis=1)

            similarityLoss = self.loss(representation1, representation2)

            Net_gradients = tape.gradient(similarityLoss, 
                                            self.base_model.trainable_variables + self.)
            

            
            # Apply the gradients to the optimizer
            self.optNet.apply_gradients(zip(Net_gradients, 
                                            self.Net.trainable_variables))


        return {"Similarity_Loss": similarityLoss}
            