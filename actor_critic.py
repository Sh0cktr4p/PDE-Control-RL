from spinup.algos.tf1.ppo import core
import tensorflow as tf

def network(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
	pass


def custom_policy(x, a, hidden_sizes, activation, output_activation, action_space):
	if isinstance(action_space, core.Box):
		act_dim = action_space.n
		logits = network(x, list(hidden_sizes)+[act_dim], activation, None)
		logp_all = tf.nn.log_softmax(logits)
		pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
		logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
		logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
		return pi, logp, logp_pi
	elif isinstance(action_space, core.Discrete):
		act_dim = a.shape.as_list()[-1]
    	mu = network(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    	log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    	std = tf.exp(log_std)
    	pi = mu + tf.random_normal(tf.shape(mu)) * std
    	logp = core.gaussian_likelihood(a, mu, log_std)
    	logp_pi = core.gaussian_likelihood(pi, mu, log_std)
    	return pi, logp, logp_pi