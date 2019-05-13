import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
from collections import OrderedDict

from yw.tool import logger
from yw.util.util import store_args

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels


class GPQEstimator:
    @store_args
    def __init__(self, scope, lr, input_dims, max_u, data_size, num_sample, **kwargs):
        """Base class for doing imitation learning.

        Args:
            # Environment I/O and Config
            input_dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
            # GP Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            # Training
            data_size          (int)          - Number of transitions to be used in the demonstration buffer
            lr                 (float)        - learning rate for the pi (actor) network

        """

        # Initialize training and test buffer
        self.observation_set = {}
        self.test_set = {}

        # Draw the computational graph
        with tf.variable_scope(self.scope):
            self._create_network()

    def get_q_value(self, transitions):

        feed = {}
        feed.update(
            {self.observation_set_tf[k]: self.observation_set[k] for k in self.observation_set_tf.keys()}
        )
        feed.update({self.input_set_tf[k]: transitions[k] for k in self.input_set_tf.keys()})

        return self.sess.run([self.output_sample_tf, self.output_mean_tf, self.output_var_tf], feed_dict=feed)

    def update_dataset(self, demo_file=None, demo_test_file=None):
        if demo_file:
            logger.info("Adding data to the training set.")
            demo_data = np.load(demo_file)  # load the demonstration data from data file
            loaded_training_set = OrderedDict(**demo_data)
            assert all([loaded_training_set[k].shape[0] >= self.data_size for k in loaded_training_set.keys()])
            self.observation_set.update(
                {k: loaded_training_set[k][0 : self.data_size, ...] for k in loaded_training_set.keys()}
            )
            for k in self.observation_set.keys():
                assert len(self.observation_set[k].shape) > 1
                logger.debug(
                    "GPImitator.update_dataset -> loaded set key {} with shape {}".format(
                        k, self.observation_set[k].shape
                    )
                )
        if demo_test_file:
            logger.info("Adding data to test set.")
            demo_data = np.load(demo_test_file)  # load the demonstration data from data file
            self.test_set.update(OrderedDict(**demo_data))
            assert all([self.test_set[k].shape[0] >= self.data_size for k in self.test_set.keys()])
            for k in self.test_set.keys():
                logger.debug(
                    "GPImitator.update_dataset -> loaded set key {} with shape {}".format(k, self.test_set[k].shape)
                )

    def check(self):
        feed = {}
        feed.update({self.observation_set_tf[k]: self.observation_set[k] for k in self.observation_set_tf.keys()})
        feed.update({self.input_set_tf[k]: self.test_set[k] for k in self.input_set_tf.keys()})
        mean, var, sample, loss = self.sess.run(
            [self.output_sample_tf, self.output_mean_tf, self.output_var_tf, self.neg_log_likelihood_tf], feed_dict=feed
        )

        logger.debug("GPImitator.check -> sample.shape {}".format(sample.shape))
        logger.debug("GPImitator.check -> mean.shape {}".format(mean.shape))
        logger.debug("GPImitator.check -> var.shape {}".format(var.shape))
        # Check the average variance
        logger.info("GPImitator.check -> average var {}".format(np.mean(var)))

        return loss

    def train(self):
        feed = {}
        feed.update({self.observation_set_tf[k]: self.observation_set[k] for k in self.observation_set_tf.keys()})
        _, loss = self.sess.run([self.optimize_op, self.neg_log_likelihood_tf], feed_dict=feed)
        return loss

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _create_network(self):
        """Function for creating all of the tensorflow graphs.
        """
        logger.info("Creating a Gaussian Process.")

        # Get a tf session
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # Prepare Input
        self.input_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith("info_"):
                continue
            self.input_shapes[key] = (None, *tuple([self.input_dims[key]]))
        self.input_shapes["q"] = (None, 1)
        logger.debug("GPImitation._create_network -> The staging shapes are: {}".format(self.input_shapes))

        # Observations
        self.observation_set_tf = OrderedDict(
            {key: tf.placeholder(tf.float64, shape=shape) for key, shape in self.input_shapes.items()}
        )
        training_data_tf = tf.concat(
            axis=1,
            values=[
                self.observation_set_tf["o"],
                self.observation_set_tf["g"],
                self.observation_set_tf["u"] / self.max_u,
            ],
        )
        training_label_tf = tf.reshape(self.observation_set_tf["q"], [-1])

        # Define a kernel with trainable parameters. Note we transform the trainable variables to apply a positivity constraint.
        amplitude_tf = tf.exp(tf.get_variable(name="amplitude", initializer=np.float64(0)))
        length_scale_tf = tf.exp(tf.get_variable(name="length_scale", initializer=np.float64(0)))
        kernel_tf = psd_kernels.ExponentiatedQuadratic(amplitude_tf, length_scale_tf)
        noise_variance_tf = tf.exp(tf.get_variable(name="noise_variance", initializer=np.float64(-5)))

        # We'll use an unconditioned GP to train the kernel parameters.
        gp = tfd.GaussianProcess(
            kernel=kernel_tf, index_points=training_data_tf, observation_noise_variance=noise_variance_tf
        )
        self.neg_log_likelihood_tf = -gp.log_prob(training_label_tf)
        self.optimize_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.99).minimize(
            self.neg_log_likelihood_tf
        )

        # New inputs
        self.input_set_tf = OrderedDict(
            {key: tf.placeholder(tf.float64, shape=shape) for key, shape in self.input_shapes.items()}
        )
        test_data_tf = tf.concat(
            axis=1, values=[self.input_set_tf["o"], self.input_set_tf["g"], self.input_set_tf["u"] / self.max_u]
        )

        # Define a gaussian regression model
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=kernel_tf,
            index_points=test_data_tf,
            observation_index_points=training_data_tf,
            observations=training_label_tf,
            observation_noise_variance=noise_variance_tf,
        )

        self.output_mean_tf = gprm.mean()
        self.output_var_tf = gprm.variance()
        self.output_sample_tf = tf.transpose(gprm.sample(self.num_sample),[1,0])

        # Initialize all variables
        tf.variables_initializer(self._global_vars()).run()

    def _vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + "/" + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/" + scope)
        return res

    def __getstate__(self):
        """
        """
        excluded_subnames = ["_tf", "_op", "sess", "_set"]  # need the observation set

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state["tf"] = self.sess.run([x for x in self._global_vars("")])
        # Add observation set explicitly
        state["observation_set"] = self.observation_set
        # Debug dump.
        logger.debug("GPQEstimator.__getstate__ -> Members: ", self.__dict__.keys())
        logger.debug("GPQEstimator.__getstate__ -> Saved members: ", state.keys())
        return state

    def __setstate__(self, state):
        self.__init__(**state)
        # load TF variables
        vars = [x for x in self._global_vars("")]
        self.observation_set.update(state["observation_set"])
        assert len(vars) == len(state["tf"])
        self.sess.run([tf.assign(var, val) for var, val in zip(vars, state["tf"])])
        # Output debug info.
        logger.debug("GPQEstimator.__getstate__ -> Loaded Params: ", state.keys())


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(0)

    data_size = 100
    num_sample = 12
    dim_o = 1
    dim_g = 1
    dim_u = 1
    test_size = 30  # along 1 dimension

    # Suppose we have some data from a known function. Note the index points in
    # general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
    # so we need to explicitly consume the feature dimensions (just the last one
    # here).
    f = lambda g, u, o: np.sin(4 * (o + g)[..., 0])  # * np.exp(-x[..., 0] ** 2)

    # Generate a training set with the above function
    observation_set = {}
    observation_set["o"] = np.random.uniform(-1.0, 1.0, [data_size, dim_o])
    observation_set["u"] = np.random.uniform(-1.0, 1.0, [data_size, dim_u])
    observation_set["g"] = np.random.uniform(-1.0, 1.0, [data_size, dim_g])
    observation_set["q"] = f(observation_set["g"], observation_set["u"], observation_set["o"])[..., np.newaxis]

    # Generate a test set with the same function
    test_set = {}
    og = np.meshgrid(np.linspace(-1.0, 1.0, test_size), np.linspace(-1.0, 1.0, test_size))
    test_set["o"] = og[0].reshape(-1, 1)
    test_set["u"] = og[1].reshape(-1, 1)
    test_set["g"] = np.linspace(-1.0, 1.0, 900).reshape(-1, 1)
    test_set["q"] = f(test_set["g"], test_set["u"], test_set["o"])[..., np.newaxis]

    gp_imitator = GPQEstimator(
        scope="test",
        lr=0.01,
        input_dims={
            "o": observation_set["o"].shape[1],
            "u": observation_set["u"].shape[1],
            "g": observation_set["g"].shape[1],
        },
        max_u=1.0,
        data_size=data_size,
        num_sample=12,
    )

    # Equivalent to gp_imitator.update_dataset(observation_set, test_set)
    gp_imitator.observation_set.update(**observation_set)
    gp_imitator.test_set.update(**test_set)

    # Train the gaussian process
    for i in range(1000):
        loss = gp_imitator.train()
        if i % 100 == 0:
            print("Step {}: NLL = {}".format(i, loss))
            gp_imitator.check()
    print("Final NLL = {}".format(loss))

    # Plot
    sample, mean, var = gp_imitator.get_q_value(test_set)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        test_set["o"].reshape(test_size, test_size),
        test_set["g"].reshape(test_size, test_size),
        test_set["q"].reshape(test_size, test_size),
        alpha=0.8,
    )
    ax.plot_surface(
        test_set["o"].reshape(test_size, test_size),
        test_set["g"].reshape(test_size, test_size),
        (mean + var).reshape(test_size, test_size),
        alpha=0.6,
    )
    ax.plot_surface(
        test_set["o"].reshape(test_size, test_size),
        test_set["g"].reshape(test_size, test_size),
        (mean - var).reshape(test_size, test_size),
        alpha=0.6,
    )
    ax.scatter(
        np.squeeze(observation_set["o"]),
        np.squeeze(observation_set["g"]),
        observation_set["q"],
        label="observed points",
        color="r",
    )
    ax.legend()
    plt.show(block=False)
    plt.pause(2)
