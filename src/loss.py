import tensorflow as tf

class WeightedBCELoss(tf.keras.losses.Loss):
    def __init__(self, w1=1.0, w2=1.0, name="weighted_bce_loss"):
        super(WeightedBCELoss, self).__init__(name=name)
        self.w1 = w1
        self.w2 = w2

    def call(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        loss = - (self.w1 * y_true * tf.math.log(y_pred) + 
                  self.w2 * (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)
