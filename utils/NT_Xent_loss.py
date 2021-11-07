import tensorflow as tf 

class NT_Xent(tf.keras.losses.Loss):
    """NT-Xent, or Normalized Temperature-scaled Cross Entropy Loss, is a loss function. 

    let sim(u, v) = u'v / ||u||v||, looks familiar? it's cosine similarity

    The loss function for a positive pair (i, j) is

    NT-Xent = -tf.math.log(tf.math.exp(sim(u, v) / t)) / 2 * sum( tf.math.exp(sim(u, v) / t))))
    """

    def __init__(self, tau, name="NT_Xent"):
        self.tau = tau




sim = tf.keras.metrics.CosineSimilarity(axis=1)
t = 5
tf.random.set_seed(0)
u = tf.random.uniform(shape = (15,15))
tf.random.set_seed(0)
v = tf.random.uniform(shape = (15, 15))

norm_cos = tf.math.exp(sim(u, v) / t)

sim = tf.keras.metrics.CosineSimilarity(axis=0)