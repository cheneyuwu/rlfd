import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

class BNN(tf.keras.Model):
    '''
    Bayesian Neural Network dynamics model

    Provides the following methods as an interface:
        __init__ - creates class instance, runs the initializers for all model weights
        call - accepts concatenated obs and act, output concatenated reward and next_obs
        get_elites - return the indices of the current elite networks (determined based on holdout loss)
        reset - re-run initialization for all weights and biases (no arguments)
        train - accepts inputs (concatenated obs and act), targets (concatenated next_obs, reward) performs of model training
        save - stores model in filesystem (TBD)
    '''

    def __init__(self, dimo, dimu, layer_sizes, weight_decay):
        super(BNN, self).__init__()
        self._input_dim = dimo + dimu #input concatenated observations and actions
        self._output_dim = (dimo + 1) * 2 #output reward and observations, mean, and variance
        self._dimo = dimo
        self._dimu = dimu
        self._weight_decays = weight_decay
        self._mlp_layers = []

        #max_log_var to 0.5
        #min_log_var to 10

        self._max_log_var = tf.Variable(
            np.ones([1, self._output_dim // 2])/2.,
            dtype=tf.float32
        )
        self._min_log_var = tf.Variable(
            -np.ones([1, self._output_dim // 2]) * 10,
            dtype=tf.float32
        )

        #build the model
        for size in layer_sizes:
            layer = tf.keras.layers.Dense(
                units = size,
                activation = "swish",
                kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=1/(2*np.sqrt(self._input_dim)))
            )
            self._mlp_layers.append(layer)
        
        #add the output layer
        self._mlp_layers.append(
            tf.keras.layers.Dense(
                units = self._output_dim,
                kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=1/(2*np.sqrt(self._input_dim)))
            )
        )

    def call(self, inputs):
        temp = inputs
        for layer in self._mlp_layers:
            temp = layer(temp)

        means, variances = temp[:,:(self._output_dim//2)], temp[:,(self._output_dim//2):]
        logvar = self._max_log_var - tf.nn.softplus(self._max_log_var - variances)
        logvar = self._min_log_var + tf.nn.softplus(logvar - self._min_log_var)
        result = tf.stack([means, logvar])
        return result
    
    def regularization_loss(self):
        #loss terms to discourage large max/min variances
        min_max_var_loss = 0.01 * tf.reduce_sum(self._max_log_var) - 0.01 * tf.reduce_sum(self._min_log_var)

        #weight decay terms for each layer
        weight_decay_losses = []
        for i, layer in enumerate(self._mlp_layers):
            weight_decay_losses.append(self._weight_decays[i] * tf.nn.l2_loss(layer.kernel))
        weight_decay_loss = tf.reduce_sum(weight_decay_losses)

        return min_max_var_loss + weight_decay_loss

    def training_loss(self, inputs, target, inc_var_loss=True): 
        results = self.call(inputs)
        means, variances = results[0], results[1]

        if inc_var_loss:
            inv_var = tf.exp(-variances)
            mse_losses = tf.reduce_mean(tf.reduce_mean(tf.square(target - means) * inv_var, axis = -1), axis=-1)
            var_losses = tf.reduce_mean(tf.reduce_mean(variances, axis=-1), axis=-1)
            return mse_losses + var_losses

        else:
            mse_losses = tf.reduce_mean(tf.reduce_mean(tf.square(target - means), axis=1), axis=-1)
            return mse_losses
    
    def inspect(self):
        print("\n Layers of the network: \n")
        for layer in self._mlp_layers:
            print(layer.get_weights())
        
        print("\n Min and max logvar: \n")
        print(self._max_log_var)
        print(self._min_log_var)

        return "\n Current state of the network"
        


class BNNEnsemble(tf.keras.Model):

    def __init__(self, dimo, dimu, layer_sizes, weight_decay, num_networks, num_elites, learning_rate=0.001):
        super(BNNEnsemble, self).__init__()
        self.mu = tf.Variable(tf.zeros(shape = tf.TensorShape([dimo + dimu])), trainable=False)
        self.sigma = tf.Variable(tf.zeros(shape = tf.TensorShape([dimo + dimu])), trainable=False)
        self._dimo = dimo
        self._output_size = (dimo + 1) * 2
        self._models = []
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._num_networks = num_networks
        self._num_elites = num_elites
        self._elite_indices = tf.Variable(tf.zeros(shape=tf.TensorShape([self._num_elites]), dtype=tf.dtypes.int32), trainable=False)
        self.holdout_losses = tf.Variable(tf.zeros(shape=tf.TensorShape([self._num_networks])), trainable=False)
        self.best_performance = tf.Variable(tf.zeros(shape=tf.TensorShape([self._num_networks])), dtype=tf.dtypes.float32, trainable=False)
        self.epochs_since_update = tf.Variable(0, dtype=tf.dtypes.int32, trainable=False)
        for _ in range(num_networks):
            self._models.append(BNN(dimo, dimu, layer_sizes, weight_decay))

    @tf.function
    def predict(self, obs, act, deterministic=False):
        inputs = (tf.concat([obs, act], 1) - self.mu) / self.sigma
        batch_size = tf.shape(inputs)[0]
        model_index = tf.random.uniform([batch_size], 0, self._num_elites, tf.dtypes.int32)
        all_means = tf.TensorArray(tf.dtypes.float32, size=self._num_networks, clear_after_read = False)
        all_logvars = tf.TensorArray(tf.dtypes.float32, size=self._num_networks, clear_after_read = False)
        for i, model in enumerate(self._models):
            result = model(inputs)
            all_means = all_means.write(i, result[0])
            all_logvars = all_logvars.write(i, result[1])
        
        means = tf.TensorArray(tf.dtypes.float32, size=batch_size, clear_after_read = False)
        logvars = tf.TensorArray(tf.dtypes.float32, size=batch_size, clear_after_read = False)
        for i in tf.range(batch_size):
            mean = tf.gather(all_means.read(self._elite_indices[model_index[i]]), i, axis=0)
            logvar = tf.gather(all_logvars.read(self._elite_indices[model_index[i]]), i, axis=0)
            means = means.write(i, mean)
            logvars = logvars.write(i, logvar)
        means = means.stack()
        logvars = logvars.stack()

        if deterministic:
            samples = means
        else:
            stds = tf.exp(-0.5 * logvars)
            samples = means + tf.random.normal(tf.shape(means)) * stds
        
        next_obs = samples[:, 1:] + obs
        rew = samples[:, :1]

        return next_obs, rew

    @tf.function
    def train(self, obs, act, next_obs, rew, batch_size=256, max_num_epochs = 100, max_epochs_since_update = 5, holdout_ratio=0.2):
        inputs = tf.concat([obs, act], 1)
        self.mu.assign(tf.reduce_mean(inputs, 0))
        self.sigma.assign(tf.math.reduce_std(inputs, 0))
        self.sigma.assign(tf.where(tf.less(self.sigma, 1e-12), tf.ones(tf.shape(self.sigma)), self.sigma))
        inputs = (inputs - self.mu) / self.sigma
        targets = tf.concat([rew, (next_obs - obs)], 1)
        num_samples = tf.cast(tf.shape(inputs)[0], tf.dtypes.float32)

        #select holdouts
        holdout_idx = tf.random.shuffle(tf.range(num_samples, dtype=tf.dtypes.int32))
        num_holdout = tf.cast(tf.math.round(holdout_ratio * num_samples), tf.dtypes.int32)

        num_samples = tf.cast(num_samples, tf.dtypes.int32)
        num_training_samples = num_samples - num_holdout
        
        holdout_idx_training = tf.slice(holdout_idx, [num_holdout], [num_training_samples])
        holdout_idx_holdout = tf.slice(holdout_idx, [0], [num_holdout])
        inputs, holdout_inputs = tf.gather(inputs, holdout_idx_training, axis=0), tf.gather(inputs, holdout_idx_holdout, axis=0)
        targets, holdout_targets = tf.gather(targets, holdout_idx_training, axis=0), tf.gather(targets, holdout_idx_holdout, axis=0)

        #create seperate training sequences (with duplicates) for inividual models
        training_idx = tf.random.uniform([self._num_networks, num_training_samples], 0, num_training_samples, tf.dtypes.int32)

        updated = False
        for i in range(max_num_epochs):
            for batch_num in tf.range(num_training_samples // batch_size):
                with tf.GradientTape() as tape:
                    total_loss = tf.reduce_sum([
                        model.training_loss(
                            tf.gather(inputs, tf.slice(training_idx[i], [batch_num * batch_size], [batch_size]), axis=0), 
                            tf.gather(targets, tf.slice(training_idx[i], [batch_num * batch_size], [batch_size]), axis=0)) 
                        for i, model in enumerate(self._models)])
                    total_loss += tf.reduce_sum([model.regularization_loss() for model in self._models])
                trainable_variables = self.trainable_variables
                gradients = tape.gradient(total_loss, trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, trainable_variables))
            #swap around the datasets between the models
            tf.random.shuffle(training_idx)
            for i in range(self._num_networks):
                self.holdout_losses[i].assign(self._models[i].training_loss(holdout_inputs, holdout_targets, False))
                best = self.best_performance[i]
                improvement = (best - self.holdout_losses[i]) / best
                improvement = tf.where(tf.is_nan(improvement), 100., improvement)
                if (improvement > 0.01):
                    self.best_performance[i] = self.holdout_losses[i]
                    updated = True
            if updated:
                self.epochs_since_update.assign(0)
            else:
                self.epochs_since_update.assign_add(1)
            if(self.epochs_since_update > max_epochs_since_update):
                break
            updated = False

        #determine elites
        for i in range(self._num_networks):
            self.holdout_losses[i].assign(self._models[i].training_loss(holdout_inputs, holdout_targets, False))

        values, indices = tf.math.top_k(-1 * self.holdout_losses, k=self._num_elites)
        self._elite_indices.assign(indices)

        #return the loss of the elites on the holdout set
        return indices, values

if __name__ == '__main__':
    # network = BNN(1, 1, [2], [0.00005, 0.00005])
    # data = tf.constant([[1., 2.], [1., 0.]])
    # target = tf.constant([[0., 0.], [0., 1.]])

    # network.inspect()
    # network(data)
    # network(data)
    # network(data)
    # network(data)
    # network(target)
    # network.inspect()
    # print(network.regularization_loss())
    # print(network.training_loss(data, target))

    model = BNNEnsemble(1, 1, [10, 10], [0.001, 0.001, 0.001], 1, 1, learning_rate=0.1)

    obs = tf.constant([[1.]])
    act = tf.constant([[5.]])
    next_obs = tf.constant([[3.]])
    rew = tf.constant([[100.]])

    model.train(obs, act, next_obs, rew, 1, 200)
    model.train(obs, act, next_obs, rew, 1, 200)
    model.train(obs, act, next_obs, rew, 1, 200)
    result = model.predict(obs, act)
    print(result)