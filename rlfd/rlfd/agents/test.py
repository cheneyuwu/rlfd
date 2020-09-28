import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

i = tf.Variable(5)
epochs_since_update = tf.Variable(0, dtype=tf.dtypes.int32, trainable=False)

@tf.function
def test(i = 5):
    for j in tf.range(i):
        if(j > 1):
            tf.print("Hello world\n")
            break
@tf.function
def train(max_num_epochs = 100, max_epochs_since_update = 5):
    print("Max num epoch: {}".format(max_num_epochs))
    for i in range(max_num_epochs):
        cond = tf.equal(epochs_since_update, max_epochs_since_update)
        tf.print(cond)
        if cond:
            break
            #continue

if __name__=="__main__":
    test()
    obs = tf.constant([[1.]])
    act = tf.constant([[5.]])
    next_obs = tf.constant([[3.]])
    rew = tf.constant([[100.]])
    train()
