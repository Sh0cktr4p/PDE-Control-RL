from spinup.algos.tf1.ppo import core
import tensorflow as tf
import numpy as np


def network(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
	print(x.shape)
	if x.shape[1:] == (64, 64, 2):
		x = tf.layers.conv2d(x, 16, (3, 3), activation=tf.nn.relu)
		x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))
		
		x = tf.layers.conv2d(x, 16, (5, 5), activation=tf.nn.relu)
		x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

		x = tf.layers.conv2d(x, 16, (5, 5), activation=tf.nn.relu)
		x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

		#x = tf.layers.conv2d(x, 4, (1, 1), activation=tf.nn.relu)
		#x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

		x = tf.layers.flatten(x)
		print(x)
		print(hidden_sizes[-1])
	
	for h in hidden_sizes[:-1]:
		x = tf.layers.dense(x, units=h, activation=activation)
	return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = network(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = network(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = core.gaussian_likelihood(a, mu, log_std)
    logp_pi = core.gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, core.Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, core.Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(network(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v
