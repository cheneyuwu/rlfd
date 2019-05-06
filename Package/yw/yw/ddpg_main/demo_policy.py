import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from yw.tool import logger
from yw.util.util import store_args
from yw.util.tf_util import nn


class EnsembleNN:
    """This class should fully encapsulate a tensorflow network for training purpose.

    Member:
        demo_samples     (list of SampleNN)
        iter_initializer (list of iter_init)
        loss_tf          (tensor)
        output_tf        (list of tensor)
        output_mean_tf   (tensor)
        output_var_tf    (tensor)
        test_tf          (tensor)
    """

    @store_args
    def __init__(self, input_tf, training_set, data_size, batch_size, num_sample, hidden, layers, max_u, **kwargs):
        """Create a bootstrapped ensemble of neural networks

        Args:
            input_tf     (dict of placeholder) - Input for testing and using the neural network.
            training_set (dict of array)       - Buffer object that stores the entire training set
            hidden       (int)                 - Number of hidden units for each layer except for the last one.
            layers       (int)                 - Number of hidden layers.
            max_u        (int)                 - Use this to make the action output between -1 and 1.

        """
        logger.info(
            "Creating an Ensemble NN of sample x hidden x layer: {} x {} x {}.".format(
                self.num_sample, self.hidden, self.layers
            )
        )
        with tf.variable_scope("EnsembleNN"):
            self.demo_samples = []
            loss_tf = []
            output_tf = []
            test_tf = []
            for i in range(self.num_sample):
                with tf.variable_scope("ens_" + str(i)):
                    demo_sample = self.SampleNN(**self.__dict__)
                    loss_tf.append(demo_sample.loss_tf)
                    output_tf.append(demo_sample.output_tf)
                    test_tf.append(demo_sample.test_tf)
                    self.demo_samples.append(demo_sample)

            self.iter_initializer = [sample.iterator_tf.initializer for sample in self.demo_samples]
            self.loss_tf = tf.reduce_mean(loss_tf)
            self.test_tf = tf.reduce_mean(test_tf)

            # method 1
            self.output_tf = tf.concat(values=output_tf, axis=1)
            self.output_mean_tf, self.output_var_tf = tf.nn.moments(self.output_tf, 1)

            # method 2 from paper simple and scalable predictive uncertainty estimation using deep ensembles
            # means = [tf.reshape(output[...,0], (-1,1)) for output in output_tf]
            # vars = [tf.reshape(output[...,1], (-1,1)) for output in output_tf]
            # self.output_tf = tf.concat(values=means, axis=1)
            # self.output_mean_tf = tf.add_n(means) / num_sample
            # self.output_var_tf = tf.add_n(vars + [ i**2 for i in means]) / num_sample - self.output_mean_tf**2


    class SampleNN:
        """One sample of the demonstration neural net.

        Member:
            training_set_tf (Dataset)
            iterator_tf     (Iterator)
            batch_tf        (tensor)
            loss_tf         (tensor)
            output_tf       (tensor)
            test_tf         (tensor)
        """

        @store_args
        def __init__(
            self, input_tf, input_shapes, training_set, data_size, batch_size, hidden, layers, max_u, **kwargs
        ):
            """Create a bootstrapped ensemble of neural networks

            Args:
                input_shapes (dict)
                training_set (dict)
                batch_tf     (dict) - Input for training. This is usually form an iterator next.
                input_tf     (dict) - Input for testing and using the neural network.
                hidden       (int)  - Number of hidden units for each layer except for the last one.
                layers       (int)  - Number of hidden layers.
                max_u        (int)  - Use this to make the action output between -1 and 1.

            """
            # Build graph for training using data in the training set
            sub_data_size = int(
                self.data_size
            )  # We can change this to a smaller number so that each sample only sees a subset of the training set

            def generate_data():
                idxs = np.random.randint(0, self.data_size, sub_data_size)
                sub_training_set = {key: self.training_set[key][idxs] for key in self.training_set.keys()}
                # sub_training_set = self.training_set
                for i in range(sub_data_size):
                    yield {key: sub_training_set[key][i] for key in self.input_shapes.keys()}

            self.training_set_tf = (
                tf.data.Dataset.from_generator(
                    generate_data,
                    output_types={k: tf.float32 for k in self.input_shapes.keys()},
                    output_shapes={key: self.input_shapes[key][1:] for key in self.input_shapes.keys()},
                )
                .take(sub_data_size)
                .shuffle(sub_data_size)
                .batch(self.batch_size)
            )

            self.iterator_tf = self.training_set_tf.make_initializable_iterator()

            # Graph for training using data in the training set
            # inputs
            self.batch_tf = self.iterator_tf.get_next()
            train_input_tf = tf.concat(
                axis=1, values=[self.batch_tf["o"], self.batch_tf["g"], self.batch_tf["u"] / self.max_u]
            )
            test_input_tf = tf.concat(
                axis=1, values=[self.input_tf["o"], self.input_tf["g"], self.input_tf["u"] / self.max_u]
            )
            # connect input to the graph

            # method 1
            train_output_tf = nn(train_input_tf, [self.hidden] * self.layers + [1])
            self.loss_tf = tf.reduce_mean(tf.square(tf.reshape(self.batch_tf["q"], [-1, 1]) - train_output_tf))
            self.output_tf = nn(test_input_tf, [self.hidden] * self.layers + [1], reuse=True)
            self.test_tf = tf.reduce_mean(tf.square(tf.reshape(self.input_tf["q"], [-1, 1]) - self.output_tf))

            # method 2 from paper simple and scalable predictive uncertainty estimation using deep ensembles
            # rescale = 0.5
            # def gaussian_nll(mean_values, var_values, y):
            #     y_diff = tf.subtract(y, mean_values)
            #     # loss = 0.5*tf.reduce_mean(tf.log(var_values)) + 0.5*tf.reduce_mean(tf.div(tf.square(y_diff), var_values)) + 0.5*tf.log(2*np.pi)
            #     loss = tf.reduce_mean(tf.log(var_values)) + tf.reduce_mean(tf.div(tf.square(y_diff), var_values))
            #     return loss
            # raw_output_tf = nn(train_input_tf, [self.hidden] * self.layers + [2])
            # output_mean_tf = tf.reshape(raw_output_tf[...,0],(-1,1)) * rescale
            # output_var_tf = tf.reshape((tf.log(1 + tf.exp(raw_output_tf[...,1])) + 1e-6) * rescale * rescale, (-1,1))
            # self.loss_tf = gaussian_nll(output_mean_tf, output_var_tf, tf.reshape(self.batch_tf["q"], [-1, 1]))

            # raw_output_tf = nn(test_input_tf, [self.hidden] * self.layers + [2], reuse=True)
            # output_mean_tf = tf.reshape(raw_output_tf[...,0],(-1,1)) * rescale
            # output_var_tf = tf.reshape((tf.log(1 + tf.exp(raw_output_tf[...,1])) + 1e-6) * rescale * rescale, (-1,1))
            # self.test_tf = gaussian_nll(output_mean_tf, output_var_tf, tf.reshape(self.input_tf["q"], [-1, 1]))
            # self.output_tf =tf.concat(axis=1, values=[output_mean_tf, output_var_tf])

            logger.debug("demo_policy.EnsembleNN.SampleNN.__init__ -> Called.")


class BaysianNN:
    @store_args
    def __init__(
        self, input_tf, training_set, data_size, batch_size, num_sample, hidden, layers, max_u, **kwargs
    ):
        """Create a baysian neural network for demonstration.

        Args:
            batch_tf   (dict) - Input for training. This is usually form an iterator next.
            input_tf   (dict) - Input for testing and using the neural network.
            num_sample (int)  - Number of samples drawn when testing and using the network
            hidden     (int)  - Number of hidden units for each layer except for the last one.
            layers     (int)  - Number of hidden layers.
            max_u      (int)  - Use this to make the action output between -1 and 1.
            data_size  (int)  - Size of the training set. This is required because the kl divergence is expected to be
                               updated once for each epoch.

        """
        with tf.variable_scope("BaysianNN"):
            # Build graph for training using data in the training set
            def generate_data():
                for i in range(self.data_size):
                    yield {key: self.training_set[key][i] for key in self.input_shapes.keys()}

            self.training_set_tf = (
                tf.data.Dataset.from_generator(
                    generate_data,
                    output_types={k: tf.float32 for k in self.input_shapes.keys()},
                    output_shapes={key: self.input_shapes[key][1:] for key in self.input_shapes.keys()},
                )
                .take(self.data_size)
                .shuffle(self.data_size)
                .batch(self.batch_size)
            )

            self.iterator_tf = self.training_set_tf.make_initializable_iterator()
            self.iter_initializer = self.iterator_tf.initializer
            # graph for training using data in the training set
            self.batch_tf = self.iterator_tf.get_next()

            nn = tf.keras.Sequential(
                [tfp.layers.DenseReparameterization(self.hidden, activation=tf.nn.relu) for _ in range(self.layers)]
                + [tfp.layers.DenseReparameterization(1)]
            )
            concat_input_tf = tf.concat(
                axis=1, values=[self.batch_tf["o"], self.batch_tf["g"], self.batch_tf["u"] / self.max_u]
            )
            train_output_tf = tfp.distributions.Normal(loc=nn(concat_input_tf), scale=0.1)
            # loss function for each instance
            # check loss function
            neg_log_likelihood = -tf.reduce_mean(
                input_tensor=train_output_tf.log_prob(tf.reshape(self.batch_tf["q"], [-1, 1]))
            )
            kl = sum(nn.losses) / self.data_size
            self.loss_tf = neg_log_likelihood + kl

            # for testing
            concat_input_tf = tf.concat(
                axis=1, values=[self.input_tf["o"], self.input_tf["g"], self.input_tf["u"] / self.max_u]
            )
            test_output = []
            for _ in range(num_sample):
                test_output.append(nn(concat_input_tf))

            self.output_tf = tf.concat(test_output, axis=1)
            self.output_mean_tf, self.output_var_tf = tf.nn.moments(self.output_tf, 1)
            output_dist_tf = tfp.distributions.Normal(loc=nn(concat_input_tf), scale=0.1)
            self.test_tf = sum(nn.losses) / self.data_size - tf.reduce_mean(
                input_tensor=output_dist_tf.log_prob(tf.reshape(self.input_tf["q"], [-1, 1]))
            )

        # Dump Information
        logger.info("Create an Baysian NN of hidden x layer: {} x {}.".format(self.hidden, self.layers))
        logger.debug("demo_policy.BaysianNN.__init__ -> TODO")


class ActorNN:
    @store_args
    def __init__(self, dimo, dimg, dimu, input_tf, training_set, data_size, batch_size, num_sample, hidden, layers, max_u, **kwargs):
        """Create a single neural network ensemble of neural networks

            The input to the network is a state and a goal.
            Out put of the network is the desired action.
            We can use this network to encode demonstration data. i.e. summarize but not memorize. Then we use this
            network to generate new experience which is more likely leading to success. May be this could help with
            exploration.
            We can then use some q filter like method to determine whether the actor is good or not.


        Args:
            batch_tf   (dict) - Input for training. This is usually form an iterator next.
            input_tf   (dict) - Input for testing and using the neural network.
            num_sample (int)  - Number of neural network instances.
            hidden     (int)  - Number of hidden units for each layer except for the last one.
            layers     (int)  - Number of hidden layers.
            max_u      (int)  - Use this to make the action output between -1 and 1.

        """
        with tf.variable_scope("BaysianNN"):
            # Build graph for training using data in the training set
            def generate_data():
                for i in range(self.data_size):
                    yield {key: self.training_set[key][i] for key in self.input_shapes.keys()}

            self.training_set_tf = (
                tf.data.Dataset.from_generator(
                    generate_data,
                    output_types={k: tf.float32 for k in self.input_shapes.keys()},
                    output_shapes={key: self.input_shapes[key][1:] for key in self.input_shapes.keys()},
                )
                .take(self.data_size)
                .shuffle(self.data_size)
                .batch(self.batch_size)
            )

            self.iterator_tf = self.training_set_tf.make_initializable_iterator()
            self.iter_initializer = self.iterator_tf.initializer
            # graph for training using data in the training set
            self.batch_tf = self.iterator_tf.get_next()

            loss_tf = []
            output_tf = []
            test_tf = []
            for i in range(self.num_sample):
                with tf.variable_scope("act_" + str(i)):
                    concat_input_tf = tf.concat(axis=1, values=[self.batch_tf["o"], self.batch_tf["g"]])
                    train_output_tf = self.max_u * tf.tanh(nn(concat_input_tf, [self.hidden] * self.layers + [self.dimu]))
                    # loss function for behavior cloning
                    loss_tf.append(
                        tf.reduce_mean(tf.square(tf.reshape(self.batch_tf["u"], [-1, self.dimu]) - train_output_tf))
                    )
                    # for testing
                    concat_input_tf = tf.concat(axis=1, values=[self.input_tf["o"], self.input_tf["g"]])
                    test_output_tf = self.max_u * nn(
                        concat_input_tf, [self.hidden] * self.layers + [self.dimu], reuse=True
                    )
                    test_tf.append(
                        tf.reduce_mean(tf.square(tf.reshape(self.input_tf["u"], [-1, self.dimu]) - test_output_tf))
                    )
                    output_tf.append(tf.reshape(test_output_tf, [-1, 1, self.dimu]))

            self.loss_tf = tf.reduce_sum(loss_tf)
            self.output_tf = tf.concat(values=output_tf, axis=1)
            self.output_mean_tf, self.output_var_tf = tf.nn.moments(self.output_tf, 1)
            self.test_tf = tf.reduce_sum(test_tf)

        # Dump Information
        logger.info("Create an Ensemble NN of hidden x layer: {} x {}.".format(self.hidden, self.layers))
        logger.debug("demo_policy.EnsembleNN.__init__ -> TODO")


if __name__ == "__main__":
    pass
