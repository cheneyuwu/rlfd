import numpy as np
import tensorflow as tf
import pickle
from collections import OrderedDict


from yw.tool import logger
from yw.ddpg_main.mpi_adam import MpiAdam
from yw.ddpg_main.demo_policy import EnsembleNN, BaysianNN, ActorNN
from yw.util.util import store_args, import_function
from yw.util.tf_util import nn, flatten_grads


class Imitator(object):
    @store_args
    def __init__(
        self,
        T,
        scope,
        lr,
        input_dims,
        hidden,
        layers,
        num_sample,
        net_type,
        max_u,
        data_size,
        batch_size,
        test_batch_size,
        reuse=False,
        **kwargs
    ):
        """Base class for doing imitation learning.

        Args:
            # Environment I/O and Config
            T                  (int)          - the time horizon for rollouts
            input_dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
            # NN Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            num_sample         (int)          - number of nns
            hidden             (int)          - number of units in the hidden layers
            layers             (int)          - number of hidden layers
            reuse              (boolean)      - whether or not the networks should be reused
            # Training
            data_size          (int)          - Number of transitions to be used in the demonstration buffer
            batch_size         (int)          - batch size for training
            test_batch_size    (int)          - batch size for testing
            lr                 (float)        - learning rate for the pi (actor) network

        """

        # Prepare parameters, sanity checks of all data file
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        # Initialize training and test buffer
        self.training_set = {}
        self.test_set = {}

        # Draw the computational graph
        with tf.variable_scope(self.scope):
            self._create_network()

    def update_dataset(self, demo_file=None, demo_test_file=None):
        if demo_file:
            logger.info("Adding data to the training set.")
            demo_data = np.load(demo_file)  # load the demonstration data from data file
            self.training_set.update(OrderedDict(**demo_data))
            for k in self.training_set.keys():
                assert self.training_set[k].shape[0] >= self.data_size
                logger.debug(
                    "Imitator.update_dataset -> loaded set key {} with shape {}".format(k, self.training_set[k].shape)
                )
        if demo_test_file:
            logger.info("Adding data to test set.")
            demo_data = np.load(demo_test_file)  # load the demonstration data from data file
            self.test_set.update(OrderedDict(**demo_data))
            for k in self.test_set.keys():
                logger.debug(
                    "Imitator.update_dataset -> loaded set key {} with shape {}".format(k, self.test_set[k].shape)
                )

    def check(self):
        transitions = self.test_set

        loss, output_mean, output_var = self.sess.run(
            [self.policy.test_tf, self.policy.output_mean_tf, self.policy.output_var_tf],
            feed_dict={self.input_tf[key]: transitions[key] for key in self.input_tf.keys()},
        )
        assert all([self.input_shapes[key][1:] == transitions[key].shape[1:] for key in self.input_shapes.keys()])
        assert output_mean.shape == output_var.shape
        # Debug dump - Recommend using test batch size == 1
        logger.debug("Imitator.check -> Input shape: ", {key: transitions[key].shape for key in transitions.keys()})
        logger.debug("Imitator.check -> Mean output shape is: {}".format(output_mean.shape))
        logger.debug("Imitator.check -> Output variance shape is: {}".format(output_var.shape))
        # value debug
        # logger.info("Imitator.check -> Mean q output is: {}".format(output_mean))
        logger.info("Imitator.check -> Mean u output is: {}".format(np.mean(output_var)))

        return loss

    def train(self):
        self.sess.run(self.policy.iter_initializer)
        iter = 0
        while True:
            try:
                grad, loss = self.sess.run([self.grad_tf, self.policy.loss_tf])
                self.adam.update(grad, self.lr)
            except tf.errors.OutOfRangeError:
                break
            iter += 1
        assert iter == np.ceil(self.data_size / self.batch_size)
        # Debug dump
        logger.debug("Imitation.train -> Number of iteration: ", iter)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _create_network(self):
        """Function for creating all of the tensorflow graphs.
        """
        logger.info("Creating a Demonstration NN.")

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
        logger.debug("Imitation.__init__ -> The staging shapes are: {}".format(self.input_shapes))
        # input to the network
        self.input_tf = OrderedDict(
            {key: tf.placeholder(tf.float32, shape=shape) for key, shape in self.input_shapes.items()}
        )
        # select which policy and loss function
        self.policy = import_function(self.net_type)(**self.__dict__)
        grads_tf = tf.gradients(self.policy.loss_tf, self._vars())
        self.grad_tf = flatten_grads(grads=grads_tf, var_list=self._vars())
        # optimizer
        self.adam = MpiAdam(self._vars(), scale_grad_by_procs=False)

        # Debug info
        logger.debug("Demo.__init__ -> Global variables are: {}".format(self._global_vars()))
        logger.debug("Demo.__init__ -> Trainable variables are: {}".format(self._vars()))

        # Initialize all variables
        tf.variables_initializer(self._global_vars()).run()
        self.adam.sync()

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
        excluded_subnames = ["_tf", "adam", "sess", "policy", "_set"]

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state["tf"] = self.sess.run([x for x in self._global_vars("") if "buffer" not in x.name])
        # Debug dump.
        logger.debug("Imitator.__getstate__ -> Members: ", self.__dict__.keys())
        logger.debug("Imitator.__getstate__ -> Saved members: ", state.keys())
        return state

    def __setstate__(self, state):
        self.__init__(**state)
        # load TF variables
        vars = [x for x in self._global_vars("") if "buffer" not in x.name]
        assert len(vars) == len(state["tf"])
        self.sess.run([tf.assign(var, val) for var, val in zip(vars, state["tf"])])
        # Output debug info.
        logger.debug("Imitator.__getstate__ -> Loaded Params: ", state.keys())


class ActionImitator(Imitator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_actions(self, o, ag, g, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):

        vals = [self.policy.output_mean_tf]
        # feed
        feed = {
            self.policy.input_tf["o"]: o.reshape(-1, self.dimo),
            self.policy.input_tf["g"]: g.reshape(-1, self.dimg),
        }
        ret = self.sess.run(vals, feed_dict=feed)

        if compute_Q:
            return ret + [0]
        else:
            return ret[0]


class QEstimator(Imitator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_q_value(self, transitions):
        return self.sess.run(
            [self.policy.output_tf, self.policy.output_mean_tf, self.policy.output_var_tf],
            feed_dict={self.input_tf[key]: transitions[key] for key in self.input_tf.keys()},
        )

    def query_output(self, filename=None):
        """Check the output from demonstration NN when the state is fixed and the action forms a 2d space.

        This check can only be used for the Reach2DFirstOrder environment.
        """
        logger.info("Query: uncertainty: Plot the uncertainty of demonstration NN.")

        transitions = self.test_set

        output_mean, output_var = self.sess.run(
            [self.policy.output_mean_tf, self.policy.output_var_tf],
            feed_dict={self.input_tf[key]: transitions[key] for key in self.input_tf.keys()},
        )
        assert all([self.input_shapes[key][1:] == transitions[key].shape[1:] for key in self.input_shapes.keys()])
        assert output_mean.shape == output_var.shape

        name = ["o", "g", "u", "q_mean", "q_var"]
        ret = [transitions["o"], transitions["g"], transitions["u"], output_mean, output_var]

        if filename:
            logger.info("Query: uncertainty: storing query results to {}".format(filename))
            np.savez_compressed(filename, **dict(zip(name, ret)))
        else:
            logger.info("Query: uncertainty: ", ret)