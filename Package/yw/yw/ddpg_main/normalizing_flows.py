import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import tensorflow as tf
tfd = tf.contrib.distributions
tfb = tfd.bijectors

batch_size=512
DTYPE=tf.float32
NP_DTYPE=np.float32

def sample_target_normal():
    mean = [0.4, 1]
    A = np.array([[2, .3], [-1., 4]])
    cov = A.T.dot(A)
    print(mean)
    print(cov)
    X = np.random.multivariate_normal(mean, cov, 2000)
    plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
    dataset = tf.data.Dataset.from_tensor_slices(X.astype(NP_DTYPE))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.prefetch(3 * batch_size)
    dataset = dataset.batch(batch_size)
    data_iterator = dataset.make_one_shot_iterator()
    x_samples = data_iterator.get_next()
    return x_samples


def sample_target_halfmoon():
    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(batch_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(batch_size, dtype=DTYPE))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)
    return x_samples


# quite easy to interpret - multiplying by alpha causes a contraction in volume.
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(
            forward_min_event_ndims=1, inverse_min_event_ndims=1, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = 1 # self._event_dims_tensor(y)  ## Note: this wil break for objects of size other than N x dim(vector)
        
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


class Flow(object):

    def __init__(self, d=2, r=2, num_layers=6):
        self.bijectors = []

        for i in range(num_layers):
            with tf.variable_scope('bijector_%d' % i):
                V = tf.get_variable('V', [d, r], dtype=DTYPE)  # factor loading
                shift = tf.get_variable('shift', [d], dtype=DTYPE)  # affine shift
                L = tf.get_variable('L', [d * (d + 1) / 2], dtype=DTYPE)  # lower triangular

                self.bijectors.append(tfb.Affine(
                    scale_tril=tfd.fill_triangular(L),
                    scale_perturb_factor=V,
                    shift=shift,
                ))
                
                alpha = tf.abs(tf.get_variable('alpha', [], dtype=DTYPE)) + .01
                self.bijectors.append(LeakyReLU(alpha=alpha))
                
        # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
        # that they are applied..
        mlp_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])), name='2d_mlp_bijector')

        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=mlp_bijector)
        

def sample_flow(base_dist, transformed_dist):
    x = base_dist.sample(512)
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(transformed_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)
        
    return x, samples, names


def visualize_flow(gs, row, samples, titles):
    X0 = samples[0]
    
    for i, j in zip([0, len(samples)-1], [0, 1]): #range(len(samples)):
        X1 = samples[i]

        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        ax = pl.subplot(gs[row, j])
        pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
        
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        

        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')

        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        pl.xlim([-10, 40])
        pl.ylim([-10, 40])
        pl.title(titles[j])
        
if __name__ == "__main__":
    
    tf.set_random_seed(0)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    x_samples = sample_target_halfmoon()
    np_samples = sess.run(x_samples)

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], DTYPE))
    transformed_dist = Flow().dist
    _, samples_no_training, _ = sample_flow(base_dist, transformed_dist)
    
    sess.run(tf.global_variables_initializer())
    samples_no_training = sess.run(samples_no_training)
    
    pl.figure()
    gs = gridspec.GridSpec(3, 3)

    # Training dataset
    ax = pl.subplot(gs[0, 0])
    pl.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color='red')
    pl.xlim([-5, 30])
    pl.ylim([-10, 10])
    pl.title('Training samples')

    # Flow before training
    visualize_flow(gs, 1, samples_no_training, ['Base dist', 'Samples w/o training'])

    loss = -tf.reduce_mean(transformed_dist.log_prob(x_samples))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess.run(tf.global_variables_initializer())

    NUM_STEPS = 100000
    global_step = []
    np_losses = []
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss])
        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(np_loss)

        if i % int(1e4) == 0:
            print(i, np_loss)

    #start = 10
    #pl.plot(np_losses[start:])

    _, samples_with_training, _ = sample_flow(base_dist, transformed_dist)
    samples_with_training = sess.run(samples_with_training)
    
    # Flow after training
    visualize_flow(gs, 2, samples_with_training, ['Base dist', 'Samples w/ training'])
    
    pl.show()

    
